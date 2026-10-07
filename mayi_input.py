"""MayI input builder.

Turns one permission request into the text the classifier reads, laid out like the inputs of
auto-1b-data:

    ### PROPOSED TOOL CALL
    tool: git
    args: {"cmd": "push --force origin feature/login"}

    ### USER REQUEST
    I rebased my feature branch, please replace the remote one.

    ### AGENT HISTORY
    [1] git({"cmd": "rebase main"})
    -> Successfully rebased and updated refs/heads/feature/login.

  Format A: the proposed tool call alone ("what does the agent want to do?").
  Format B: the tool call, then the user's request and the agent's earlier steps.

The tool call always comes first, so if a tokenizer truncates a long input it cuts the history,
never the call. Rows from other sources can also carry CONTEXT and RULES sections; auto-1b-data
rows never have them.
"""

import json
import re

FORMATS = ("A", "B")
NO_HISTORY = "(no prior actions)"


def _text(value):
    """Strip a value; return "" for None/NaN."""
    if value is None or (isinstance(value, float) and value != value):  # NaN from pandas
        return ""
    return str(value).strip()


def _json(value):
    """Parse a JSON string; leave anything else (dicts, lists, plain text) as it is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _args(value):
    """Arguments as they appear in the data: auto-1b-data stores them as a ready string."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return "" if value is None else str(value)


def split_call(call):
    """(tool, args) from a {"tool", "args"} dict, its JSON string, or text like "rm(path='/tmp')"."""
    call = _json(call)
    if isinstance(call, dict):
        return _text(call.get("tool") or call.get("name")), _args(call.get("args", call.get("arguments")))
    text = _text(call)
    match = re.match(r"^([\w.\-]+)\((.*)\)$", text, re.DOTALL)
    return (match.group(1), match.group(2)) if match else (text, "")


def history_lines(history):
    """Numbered steps from a list of {"tool", "args", "result"}, its JSON string, or plain text."""
    history = _json(history)
    if not history:
        return NO_HISTORY
    if isinstance(history, str):
        return history.strip()
    lines = []
    for i, step in enumerate(history, start=1):
        if isinstance(step, dict):
            lines.append(f"[{i}] {_text(step.get('tool'))}({_args(step.get('args'))})")
            lines.append(f"-> {_args(step.get('result'))}")  # results kept verbatim, like the data
        else:
            lines.append(f"[{i}] {_text(step)}")
    return "\n".join(lines)


def _rules(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split("|")
    return [r for r in (_text(v) for v in value) if r]


def build_input(example, fmt="A"):
    """Build the model input for one example.

    `example` is a dict with "call" ({"tool", "args"}, its JSON string, or text like
    "send_email(to='x')"); "user_request", "history", "context" and "rules" are optional.
    """
    tool, args = split_call(example.get("call"))
    if not tool:
        raise ValueError("example has no tool call")
    call = f"### PROPOSED TOOL CALL\ntool: {tool}\nargs: {args}"
    if fmt == "A":
        return call
    if fmt == "B":
        sections = [call, f"### USER REQUEST\n{_text(example.get('user_request'))}",
                    f"### AGENT HISTORY\n{history_lines(example.get('history'))}"]
        if _text(example.get("context")):
            sections.append(f"### CONTEXT\n{_text(example.get('context'))}")
        rules = _rules(example.get("rules"))
        if rules:
            sections.append("### RULES\n" + "\n".join(f"- {r}" for r in rules))
        return "\n\n".join(sections)
    raise ValueError(f"unknown format {fmt!r}; use one of {FORMATS}")
