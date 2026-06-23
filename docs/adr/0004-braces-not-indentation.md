# Indentation-sensitive syntax, not braces

malus uses Python-style indentation for block delimiters. Function and kernel signatures end with `:` and the body is indented; dedent signals the end of the block. This was a deliberate reversal of an earlier brace-based decision after the target users (ML researchers) expressed a strong preference for Python's visual structure.

**Lexer approach:** INDENT and DEDENT are emitted as synthetic tokens by the lexer (the same approach CPython uses), keeping the parser itself unaware of raw whitespace. This is the industry-standard solution for indentation-sensitive grammars.

## REPL consideration

Indentation-sensitive parsing requires the REPL to detect when a block is complete. The chosen convention: a **blank line** signals end-of-block at the REPL prompt, matching Python's `>>>` behavior. Multi-line editing is best handled in the Jupyter kernel (a priority v1 follow-up), which manages cell boundaries explicitly.

## Considered Options

- **Braces (`{}`)**: Originally chosen for REPL robustness and paste-friendliness. Rejected after researcher feedback — the target users' familiarity with Python indentation outweighs the parser complexity cost.
- **Indentation with INDENT/DEDENT tokens**: Chosen. Well-understood technique, directly maps to user expectations.
