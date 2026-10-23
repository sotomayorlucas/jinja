"""The optimizer tries to constant fold expressions and modify the AST
in place so that it should be faster to evaluate.

Because the AST does not contain all the scoping information and the
compiler has to find that out, we cannot do all the optimizations we
want. For example, loop unrolling doesn't work because unrolled loops
would have a different scope. The solution would be a second syntax tree
that stored the scoping rules.
"""

import typing as t

from . import nodes
from .visitor import NodeTransformer

if t.TYPE_CHECKING:
    from .environment import Environment


def optimize(node: nodes.Node, environment: "Environment") -> nodes.Node:
    """The context hint can be used to perform an static optimization
    based on the context given."""
    optimizer = Optimizer(environment)
    return t.cast(nodes.Node, optimizer.visit(node))


class Optimizer(NodeTransformer):
    def __init__(self, environment: "Environment | None") -> None:
        self.environment = environment
        # Expressions whose ``as_const()`` succeeded but whose value could
        # not be stored in a ``Const`` node (for example a generator
        # returned by ``map``). A parent can still fold them, so they count
        # as constant operands. Keyed by ``id`` and holding a reference so
        # the id can't be reused while the optimizer is alive.
        self._unwrapped: dict[int, nodes.Expr] = {}

    def _is_const(self, node: nodes.Node | None) -> bool:
        return isinstance(node, nodes.Const) or id(node) in self._unwrapped

    def _may_fold(self, node: nodes.Expr) -> bool:
        """Cheap check for whether ``node.as_const()`` can succeed.

        Folding happens bottom up, so every operand that can be folded
        has already been replaced by a ``Const`` (or recorded in
        ``_unwrapped``). If an operand that ``as_const()`` evaluates is
        anything else, its ``as_const()`` already raised ``Impossible``
        and will raise it again, but only after walking that operand's
        whole subtree. Doing that at every node of a long expression
        makes folding cubic in the length of the expression, so a
        template of a few hundred bytes can take seconds to compile.

        This returns ``False`` only when ``as_const()`` is certain to
        raise ``Impossible``, so it never changes what gets folded.
        """
        if isinstance(node, (nodes.And, nodes.Or)):
            if not self._is_const(node.left):
                return False

            if not isinstance(node.left, nodes.Const):
                return True

            # ``and`` stops at a falsy left side, ``or`` at a truthy one,
            # without evaluating the right side.
            if bool(node.left.value) is isinstance(node, nodes.Or):
                return True

            return self._is_const(node.right)

        if isinstance(node, nodes.CondExpr):
            if not self._is_const(node.test):
                return False

            if not isinstance(node.test, nodes.Const):
                return True

            if node.test.value:
                return self._is_const(node.expr1)

            return node.expr2 is not None and self._is_const(node.expr2)

        if isinstance(node, nodes.Compare):
            return self._compare_may_fold(node)

        for child in node.iter_child_nodes():
            if isinstance(child, nodes.Expr):
                if not self._is_const(child):
                    return False
            elif isinstance(child, nodes.Helper):
                # Pair and Keyword hold the expressions that as_const()
                # evaluates, so look one level further down.
                for sub in child.iter_child_nodes():
                    if isinstance(sub, nodes.Expr) and not self._is_const(sub):
                        return False

        return True

    def _compare_may_fold(self, node: nodes.Compare) -> bool:
        # Compare.as_const() stops at the first false comparison, so
        # operands after it are never evaluated.
        if not self._is_const(node.expr):
            return False

        if not isinstance(node.expr, nodes.Const):
            return True

        value = node.expr.value

        for op in node.ops:
            if not self._is_const(op.expr):
                return False

            if not isinstance(op.expr, nodes.Const):
                return True

            try:
                result = nodes._cmpop_to_func[op.op](value, op.expr.value)
            except Exception:
                # as_const() raises Impossible here without evaluating the
                # remaining operands.
                return True

            if not result:
                return True

            value = op.expr.value

        return True

    def generic_visit(
        self, node: nodes.Node, *args: t.Any, **kwargs: t.Any
    ) -> nodes.Node:
        node = super().generic_visit(node, *args, **kwargs)

        # Do constant folding. Some other nodes besides Expr have
        # as_const, but folding them causes errors later on.
        if isinstance(node, nodes.Expr) and self._may_fold(node):
            try:
                value = node.as_const(args[0] if args else None)
            except nodes.Impossible:
                return node

            try:
                return nodes.Const.from_untrusted(
                    value, lineno=node.lineno, environment=self.environment
                )
            except nodes.Impossible:
                self._unwrapped[id(node)] = node

        return node
