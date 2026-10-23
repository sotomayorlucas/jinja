import os
import re

import pytest

from jinja2 import UndefinedError
from jinja2.environment import Environment
from jinja2.loaders import DictLoader


def test_filters_deterministic(tmp_path):
    src = "".join(f"{{{{ {i}|filter{i} }}}}" for i in range(10))
    env = Environment(loader=DictLoader({"foo": src}))
    env.filters.update(dict.fromkeys((f"filter{i}" for i in range(10)), lambda: None))
    env.compile_templates(tmp_path, zip=None)
    name = os.listdir(tmp_path)[0]
    content = (tmp_path / name).read_text("utf8")
    expect = [f"filters['filter{i}']" for i in range(10)]
    found = re.findall(r"filters\['filter\d']", content)
    assert found == expect


def test_import_as_with_context_deterministic(tmp_path):
    src = "\n".join(f'{{% import "bar" as bar{i} with context %}}' for i in range(10))
    env = Environment(loader=DictLoader({"foo": src}))
    env.compile_templates(tmp_path, zip=None)
    name = os.listdir(tmp_path)[0]
    content = (tmp_path / name).read_text("utf8")
    expect = [f"'bar{i}': " for i in range(10)]
    found = re.findall(r"'bar\d': ", content)[:10]
    assert found == expect


def test_top_level_set_vars_unpacking_deterministic(tmp_path):
    src = "\n".join(f"{{% set a{i}, b{i}, c{i} = tuple_var{i} %}}" for i in range(10))
    env = Environment(loader=DictLoader({"foo": src}))
    env.compile_templates(tmp_path, zip=None)
    name = os.listdir(tmp_path)[0]
    content = (tmp_path / name).read_text("utf8")
    expect = [
        f"context.vars.update({{'a{i}': l_0_a{i}, 'b{i}': l_0_b{i}, 'c{i}': l_0_c{i}}})"
        for i in range(10)
    ]
    found = re.findall(
        r"context\.vars\.update\(\{'a\d': l_0_a\d, 'b\d': l_0_b\d, 'c\d': l_0_c\d\}\)",
        content,
    )[:10]
    assert found == expect
    expect = [
        f"context.exported_vars.update(('a{i}', 'b{i}', 'c{i}'))" for i in range(10)
    ]
    found = re.findall(
        r"context\.exported_vars\.update\(\('a\d', 'b\d', 'c\d'\)\)",
        content,
    )[:10]
    assert found == expect


def test_loop_set_vars_unpacking_deterministic(tmp_path):
    src = "\n".join(f"  {{% set a{i}, b{i}, c{i} = tuple_var{i} %}}" for i in range(10))
    src = f"{{% for i in seq %}}\n{src}\n{{% endfor %}}"
    env = Environment(loader=DictLoader({"foo": src}))
    env.compile_templates(tmp_path, zip=None)
    name = os.listdir(tmp_path)[0]
    content = (tmp_path / name).read_text("utf8")
    expect = [
        f"_loop_vars.update({{'a{i}': l_1_a{i}, 'b{i}': l_1_b{i}, 'c{i}': l_1_c{i}}})"
        for i in range(10)
    ]
    found = re.findall(
        r"_loop_vars\.update\(\{'a\d': l_1_a\d, 'b\d': l_1_b\d, 'c\d': l_1_c\d\}\)",
        content,
    )[:10]
    assert found == expect


def test_block_set_vars_unpacking_deterministic(tmp_path):
    src = "\n".join(f"  {{% set a{i}, b{i}, c{i} = tuple_var{i} %}}" for i in range(10))
    src = f"{{% block test %}}\n{src}\n{{% endblock test %}}"
    env = Environment(loader=DictLoader({"foo": src}))
    env.compile_templates(tmp_path, zip=None)
    name = os.listdir(tmp_path)[0]
    content = (tmp_path / name).read_text("utf8")
    expect = [
        f"_block_vars.update({{'a{i}': l_0_a{i}, 'b{i}': l_0_b{i}, 'c{i}': l_0_c{i}}})"
        for i in range(10)
    ]
    found = re.findall(
        r"_block_vars\.update\(\{'a\d': l_0_a\d, 'b\d': l_0_b\d, 'c\d': l_0_c\d\}\)",
        content,
    )[:10]
    assert found == expect


def test_undefined_import_curly_name():
    env = Environment(
        loader=DictLoader(
            {
                "{bad}": "{% from 'macro' import m %}{{ m() }}",
                "macro": "",
            }
        )
    )

    # Must not raise `NameError: 'bad' is not defined`, as that would indicate
    # that `{bad}` is being interpreted as an f-string. It must be escaped.
    with pytest.raises(UndefinedError):
        env.get_template("{bad}").render()


def test_constant_folding_does_not_rewalk_subtrees(monkeypatch):
    # Folding used to call as_const() on every node of a long expression,
    # and each call walked the whole subtree below it again, which made
    # compiling ``x + 1 + 1 + ...`` cubic in the length of the chain.
    from jinja2 import nodes

    calls = 0
    original = nodes.BinExpr.as_const

    def counting_as_const(self, eval_ctx=None):
        nonlocal calls
        calls += 1
        return original(self, eval_ctx)

    monkeypatch.setattr(nodes.BinExpr, "as_const", counting_as_const)
    n = 150
    # raw=True: the generated Python would exceed CPython's nesting limit.
    Environment().compile("{{ " + " + ".join(["x"] + ["1"] * n) + " }}", raw=True)
    assert calls < n * n


@pytest.mark.parametrize(
    ("source", "expect"),
    [
        ("{{ 0 and x }}", "0"),
        ("{{ 1 or x }}", "1"),
        ("{{ 'a' if 1 else x }}", "a"),
        ("{{ x if 0 else 'b' }}", "b"),
        ("{{ 1 > 2 < x }}", "False"),
    ],
)
def test_constant_folding_short_circuit(source, expect):
    # Operands that are never evaluated don't prevent folding.
    env = Environment()
    code = env.compile(source, raw=True)
    assert f"yield {expect!r}" in code
    assert env.from_string(source).render() == expect
