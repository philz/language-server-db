#!/usr/bin/env python3
#
# Given a language server and a list of file, populates a DuckDB database
# with data from the language server. Sample invocation:
#
# ~/venv/bin/python3 language-server-db.py db --db db.db \
#   --workspace ~/src/hono \
#   --file-of-files-to-process <(find ~/src/hono/src/ -name '*.ts') \
#   --lsp foo/node_modules/.bin/typescript-language-server  --stdio
#
# The language server comes from "npm i typescript-language-server".
#
# For TypeScript, the "real" language server is something called "tsserver" which is historically
# a precursor to LSPs. (There's some history at https://en.wikipedia.org/wiki/Language_Server_Protocol.)
# The typescript-language-server NPM package translates between LSP and tsserver.
# https://github.com/yioneko/vtsls is another similar project.
#
# This was written to learn some bits about language servers and to support
# a blog post; caveat emptor.
#
# Ideas for elaboration:
#  * Try with other language servers.
#  * Collect diagnostics, problems, references, definitions, etc.
#  * Visualize symbols and tokens.

from collections import defaultdict
import time
import logging
import subprocess
import re
import json
import subprocess
import os
import argparse

# These are not part of the standard library:
import duckdb
from jinja2 import Environment
import markdown
import duckdb

CONTENT_LENGTH_PATTERN = re.compile(r"^Content-Length: (\d+)\r\n\r\n$")

# See https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_documentSymbol
SYMBOL_KIND = {
    1: "File",
    2: "Module",
    3: "Namespace",
    4: "Package",
    5: "Class",
    6: "Method",
    7: "Property",
    8: "Field",
    9: "Constructor",
    10: "Enum",
    11: "Interface",
    12: "Function",
    13: "Variable",
    14: "Constant",
    15: "String",
    16: "Number",
    17: "Boolean",
    18: "Array",
    19: "Object",
    20: "Key",
    21: "Null",
    22: "EnumMember",
    23: "Struct",
    24: "Event",
    25: "Operator",
    26: "TypeParameter",
}


class LanguageServerClient:
    """Manages the language server child process and sending it requests."""

    def __init__(self, cmd, verbose):
        self.sequence = 0
        self.cmd = cmd
        self.verbose = verbose

    def start(self):
        env = os.environ.copy()
        if self.verbose:
            env["TSS_LOG"] = "-level verbose -file /tmp/tss.log"
        self.process = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
            env=env,
        )

    @staticmethod
    def _read(pipe):
        """
        Language servers send a content-length followed by a JSON payload.
        """
        while True:
            line = pipe.readline() + pipe.readline()
            logging.debug(f"Read: {line}")
            content_length = CONTENT_LENGTH_PATTERN.match(line.decode()).group(1)
            content_length = int(content_length)

            # pipe.read(n) can return fewer than n bytes
            data = b""
            while len(data) < content_length:
                data += pipe.read(content_length - len(data))
            ret = json.loads(data)

            # We can get notifications at any point; we more or less ignore them,
            # which is why we're in a while True here.
            if "method" in ret and ret["method"] == "window/logMessage":
                logging.debug(ret)
            else:
                logging.debug(ret)
                return ret

    # Spec at https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/
    def request(self, method, params):
        """Requests expect a response back."""
        self.sequence += 1
        d = dict(
            jsonrpc="2.0",
            id=self.sequence,
            type="request",
            method=method,
            params=params,
        )
        djson = json.dumps(d)
        content_length = len(djson)
        logging.debug(f"Requesting: {djson}")
        self.process.stdin.write(
            f"Content-Length: {content_length + 1}\r\n\r\n{djson}\n".encode()
        )
        self.process.stdin.flush()
        ret = self._read(self.process.stdout)
        if "error" in ret:
            raise Exception(ret["error"])
        return ret["result"]

    def notify(self, method, params):
        """Notifications don't expect anything back."""
        d = dict(jsonrpc="2.0", method=method, params=params)
        djson = json.dumps(d)
        content_length = len(djson)
        self.process.stdin.write(
            f"Content-Length: {content_length + 1}\r\n\r\n{djson}\n".encode()
        )
        self.process.stdin.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Extract language server data into a DuckDB database."
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to be verbose.",
    )
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    parser_db = subparsers.add_parser("db", help="Generate a DuckDB database")
    parser_db.add_argument(
        "--db", required=True, type=str, help="DuckDB file to create."
    )
    parser_db.add_argument(
        "--workspace", required=True, type=str, help="Workspace directory."
    )
    parser_db.add_argument(
        "--file-of-files-to-process",
        type=str,
        help="A new-line delimited list of files to process.",
    )
    parser_db.add_argument(
        "--lsp", nargs=argparse.REMAINDER, help="The language server entrypoint."
    )

    parser_html = subparsers.add_parser("html", help="Generate HTML output")
    parser_html.add_argument(
        "--db", required=True, type=str, help="DuckDB file to read from."
    )
    parser_html.add_argument(
        "--output", required=True, type=str, help="HTML output file."
    )

    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(level=level)

    if args.command == "db":
        db(args)
    elif args.command == "html":
        html(args)
    else:
        assert False


def html(args):
    # Connect to the DuckDB database
    conn = duckdb.connect(args.db)
    cursor = conn.cursor()

    # Fetch the data from the database with grouping and aggregation
    cursor.execute(
        """
        WITH t AS (
            SELECT
                s.uri,
                s.path,
                s.line,
                s.content,
                ARRAY_AGG(hover_value ORDER BY hover_start_character) FILTER (hover_value IS NOT NULL) as hover_values,
                ARRAY_AGG(hover_start_character ORDER BY hover_start_character) FILTER (hover_value IS NOT NULL) as hover_start_characters,
                ARRAY_AGG(hover_end_character ORDER BY hover_start_character) FILTER (hover_value IS NOT NULL) as hover_end_characters,
            FROM source s
            LEFT JOIN hovers hov ON s.uri = hov.uri AND s.line = hov.hover_start_line
            GROUP BY 1, 2, 3, 4
            ORDER BY s.uri, s.path, s.line
        ), t2 AS (
            SELECT t.*,
                    ARRAY_AGG(symbol_name) FILTER (symbol_name IS NOT NULL) as symbol_names,
                    ARRAY_AGG(symbol_kind) FILTER (symbol_name IS NOT NULL) as symbol_kinds,
                    ARRAY_AGG(symbol_start_character) FILTER (symbol_name IS NOT NULL) as symbol_start_characters
            FROM t
            LEFT JOIN symbols ON t.uri = symbols.uri AND t.line = symbols.symbol_start_line
            GROUP BY 1, 2, 3, 4, 5, 6, 7 ORDER BY 1, 2, 3
        )
        SELECT t2.*,
            ARRAY_AGG(token_type) FILTER (token_type IS NOT NULL) as token_types,
            ARRAY_AGG(start) FILTER (token_type IS NOT NULL) as token_starts,
            ARRAY_AGG(length) FILTER (token_type IS NOT NULL) as token_lengths,
            ARRAY_AGG(modifiers) FILTER (token_type IS NOT NULL) as token_modifiers
        FROM t2
        LEFT JOIN tokens ON t2.uri = tokens.uri AND t2.line = tokens.line
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ORDER BY 1, 2, 3;

    """
    )
    data = cursor.fetchall()
    path_by_uri = dict()
    by_file = defaultdict(list)
    for row in data:
        # Putting together lines is a bit tricky. We want
        # Something like: "while <span class="annotated">True<span class="annotation">That</span></span>:"
        # Let's hope the annotations (the hovers) aren't overlapping, and put things together based on
        # their ordering.
        result = []
        tail = 0
        # Symbols and Tokens are also there, but not used yet.
        uri, path, line, content, values, start_characters, end_characters = row[0:7]
        path_by_uri[uri] = path
        if values is None:
            annotations = []
        else:
            annotations = list(zip(values, start_characters, end_characters))
        for idx, (hover_value, hover_start, hover_end) in enumerate(annotations):
            # Technically should check that the thing is markdown, and do language-specific
            # highlighting...
            hover_value2 = markdown.markdown(
                hover_value.replace("```typescript", "```")
            )
            if hover_start > tail:
                result.append((content[tail:hover_start], None))
                tail = hover_start
            # We extend the "tail" to the next annotation if there is one or the
            # end of the line. This is largely visual, so that the annotation can extend
            # farther to the right.
            tail = hover_end
            extra = ""
            if idx + 1 < len(annotations):
                assert tail < start_characters[idx + 1]
                tail = start_characters[idx + 1] - 1
            else:
                tail = len(content)
                # We want our annotations/glosses to extend farther to the right
                # at the end of lines; I'm sure there are cleaner ways to extend
                # the element.
                extra = "&nbsp;" * 10
            result.append((content[hover_start:tail] + extra, hover_value2))
        if tail < len(content):
            result.append((content[tail:], None))
        by_file[uri].append(result)

    # Create a Jinja2 environment and load the template
    env = Environment()
    template = env.from_string(
        """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Source Files with Annotations</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                background-color: #f0f0f0;
            }
            details summary {
                padding: 40px;
                padding-left: 120px;
                font-size: 20px;
                background-color: lightblue;
            }
            .code-container {
                background-color: white;
                padding: 20px;
                counter-reset: line;
            }
            pre {
                font-family: 'Courier New', Courier, monospace;
                font-size: 16px;
                position: relative;
                line-height: 4;
            }
            .annotated {
                position: relative;
                display: inline-block;
            }
            .annotation {
                line-height: 1.2;
                position: absolute;
                font-family: Arial, sans-serif;
                font-size: 8px;
                white-space: nowrap;
                text-wrap: auto;
                left: 0;
                max-width: 100%;
                color: green;
                overflow: hidden;
                bottom: 40px;
                max-height: 55px;
            }
            .annotation:hover {
                overflow: visible;
            }
            span.linen::before {
                counter-increment: line;
                content: counter(line);
                position: relative;
                left: -3em;
                width: 2em;
                /* I don't know why this alignment isn't working. */
                text-align: right;
                color: #888;
            }
        }
        </style>
    </head>
    <body>
        {% for file, lines in by_file.items() %}
            <details>
            <summary>{{ path_by_uri[file] }}</summary>
            <div class="code-container">
            <pre>
            {% for line in lines %}
                <span class="line"><span class="linen"></span>{% for text, annotation in line %}<span class="annotated">{{ text | safe }}{% if annotation is not none %}<span class="annotation">{{ annotation | safe }}</span>{% endif %}</span>{%- endfor %}</span>
            {%- endfor %}
            </pre>
            </div>
            </details>
        {% endfor %}
    </body>
    </html>
    """
    )

    # Render the template with the data
    html_output = template.render(
        by_file=by_file, path_by_uri=path_by_uri, enumerate=enumerate
    )

    # Write the HTML output to a file
    with open("output.html", "w", encoding="utf-8") as f:
        f.write(html_output)


def db(args):
    start_time = time.time()
    ls = LanguageServerClient(args.lsp, verbose=args.verbose)
    ls.start()
    initialize_response = ls.request(
        "initialize",
        {
            "processId": os.getpid(),
            "workspaceFolders": [
                {"uri": f"file://{args.workspace}", "name": "workspace"}
            ],
            "trace": "verbose",
            # If you don't specify capabilities, you get an obscure "Request initialize failed with message: Cannot read properties of undefined (reading 'workspace')" error.
            "capabilities": {},
            "initializationOptions": {
                "logDirectory": "/tmp",
                "logVerbosity": "verbose",
                "trace": "verbose",
            },
        },
    )
    # (Pdb) initialize_response['capabilities']['semanticTokensProvider']
    # {'documentSelector': None, ['declaration', 'static', 'async', 'readonly', 'defaultLibrary', 'local']}, 'full': True, 'range': True}
    legend = initialize_response["capabilities"]["semanticTokensProvider"]["legend"]
    token_types = legend["tokenTypes"]
    token_modifiers = legend["tokenModifiers"]

    # Open a DuckDB connection
    conn = duckdb.connect(args.db)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE symbols (
            uri VARCHAR NOT NULL,
            symbol_name VARCHAR NOT NULL,
            symbol_kind VARCHAR NOT NULL,
            symbol_start_line INTEGER NOT NULL,
            symbol_start_character INTEGER NOT NULL,
            symbol_end_line INTEGER NOT NULL,
            symbol_end_character INTEGER NOT NULL,
            symbol_container_name VARCHAR
        );
        CREATE TABLE hovers (
                   uri VARCHAR NOT NULL,
                   hover_kind VARCHAR NOT NULL,
                   hover_start_line INTEGER NOT NULL,
                   hover_start_character INTEGER NOT NULL,
                   hover_end_line INTEGER NOT NULL,
                   hover_end_character INTEGER NOT NULL,
                   hover_value VARCHAR NOT NULL
        );
        CREATE TABLE source (
            uri VARCHAR NOT NULL,
            path VARCHAR NOT NULL, -- Relative to workspace; better for display.
            line INTEGER NOT NULL,
            content VARCHAR NOT NULL
        );
        CREATE TABLE tokens (
            uri VARCHAR NOT NULL,
            line INTEGER NOT NULL,
            start INTEGER NOT NULL,
            length INTEGER NOT NULL,
            token_type VARCHAR NOT NULL,
            modifiers VARCHAR[] NOT NULL,
            content VARCHAR NOT NULL
        );
    """
    )

    # It's subtle, but "didOpen" is a notification whereas "hover" is a request.
    for file in open(args.file_of_files_to_process).readlines():
        file = file.strip()
        file_uri = f"file://{file}"
        file_contents = open(file).read()
        file_lines = file_contents.splitlines()
        ls.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": file_uri,
                    "languageId": "typescript",
                    "version": 1,
                    "text": file_contents,
                },
            },
        )
        for idx, line in enumerate(file_lines):
            query = (
                """INSERT INTO source (uri, path, line, content) VALUES (?, ?, ?, ?)"""
            )
            path = os.path.relpath(file, args.workspace)
            cursor.execute(query, (file_uri, path, idx, line))

        tokens_response = ls.request(
            "textDocument/semanticTokens/full", {"textDocument": {"uri": file_uri}}
        )

        line = 0
        start = 0
        # The representation of tokens is explained in https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_semanticTokens
        # under the section "Integer Encoding for Tokens"
        #
        # For simplicity, we're going to assume multilineTokenSupport and overlappingTokenSupport
        # are both not happening.

        # itertools.batched is too new, so we implement it:
        batch = zip(
            *([iter(tokens_response["data"])] * 5)
        )  # itertools.batched(tokens_response['data'], 5)
        for deltaLine, deltaStart, length, tokenType, tokenModifiers in batch:
            query = """INSERT INTO tokens (uri, line, start, length, token_type, modifiers, content)
            VALUES (?, ?, ?, ?, ?, ?, ?)"""
            line += deltaLine
            if deltaLine == 0:
                start += deltaStart
            else:
                start = deltaStart
            modifiers = [
                token_modifiers[m]
                for m in range(tokenModifiers.bit_length())
                if tokenModifiers & (1 << m)
            ]
            content = file_lines[line][start : start + length]
            assert len(content) == length
            cursor.execute(
                query,
                (
                    file_uri,
                    line,
                    start,
                    length,
                    token_types[tokenType],
                    modifiers,
                    content,
                ),
            )

            # Let's take the hover for every semantic token!
            hover_response = ls.request(
                "textDocument/hover",
                {
                    "textDocument": {"uri": file_uri},
                    "position": dict(line=line, character=start),
                },
            )
            if hover_response is not None:
                hover_kind = (hover_response["contents"]["kind"],)
                hover_value = (hover_response["contents"]["value"],)
                # Technically you can have many!
                assert len(hover_value) == 1
                hover_value = hover_value[0]
                hover_range = hover_response["range"]
                hover_start_line = hover_range["start"]["line"]
                hover_start_character = hover_range["start"]["character"]
                hover_end_line = hover_range["end"]["line"]
                hover_end_character = hover_range["end"]["character"]

                query = """
                    INSERT INTO hovers (uri, hover_kind, hover_value, hover_start_line,
                        hover_start_character, hover_end_line, hover_end_character)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(
                    query,
                    (
                        file_uri,
                        hover_kind,
                        hover_value,
                        hover_start_line,
                        hover_start_character,
                        hover_end_line,
                        hover_end_character,
                    ),
                )

        symbols_response = ls.request(
            "textDocument/documentSymbol", {"textDocument": {"uri": f"file://{file}"}}
        )
        for item in symbols_response:
            query = """
                INSERT INTO symbols (uri, symbol_name, symbol_kind,
                    symbol_start_line, symbol_start_character, symbol_end_line, symbol_end_character,
                    symbol_container_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(
                query,
                (
                    item["location"]["uri"],
                    item["name"],
                    SYMBOL_KIND.get(item["kind"]),
                    item["location"]["range"]["start"]["line"],
                    item["location"]["range"]["start"]["character"],
                    item["location"]["range"]["end"]["line"],
                    item["location"]["range"]["end"]["character"],
                    item.get("containerName"),
                ),
            )

        ls.notify(
            "textDocument/didClose",
            {
                "textDocument": {
                    "uri": file_uri,
                },
            },
        )
    logging.info(f"Done after {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    main()
