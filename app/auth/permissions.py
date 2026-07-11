"""Role -> allowed tools/domains mapping, enforced once here rather than
re-implemented in every tool/domain. A customer-facing bot should never
see known-issues internal bug docs, for example.

User -> role resolution is a placeholder in-memory map; swap `get_role`
for a real identity/directory lookup when one exists — nothing else in
the codebase needs to change since callers only ever see a Role.
"""
from dataclasses import dataclass
from typing import Any

_ALL_DOMAINS = frozenset({"compliance", "support-docs", "known-issues"})


@dataclass(frozen=True)
class Role:
    name: str
    allowed_domains: frozenset[str]
    allowed_tools: frozenset[str] | None  # None = all tools allowed


ROLES: dict[str, Role] = {
    "customer": Role(
        name="customer",
        # compliance + support-docs are customer-facing; known-issues is
        # the internal bug tracker and should never reach a customer bot.
        allowed_domains=frozenset({"compliance", "support-docs"}),
        # Read-only status checks on a customer's own shipments/orders/
        # returns only — never inbound/inventory (internal ops) and never
        # a needs_approval tool.
        allowed_tools=frozenset({"getShipmentStatus", "getSalesOrderStatus", "getReturnStatus"}),
    ),
    "support_agent": Role(
        name="support_agent",
        allowed_domains=_ALL_DOMAINS,
        # All read-only tools across every domain; write actions
        # (updateShipmentEta, receiveAsn, allocateInventory, createPickList,
        # adjustStock, approveReturn) stay manager-only.
        allowed_tools=frozenset(
            {
                "getShipmentStatus",
                "listShipments",
                "getAsnStatus",
                "listAsns",
                "getSalesOrderStatus",
                "listSalesOrders",
                "getStockLevel",
                "listLowStock",
                "getReturnStatus",
                "listReturns",
            }
        ),
    ),
    "manager": Role(
        name="manager",
        allowed_domains=_ALL_DOMAINS,
        allowed_tools=None,
    ),
}

# Placeholder user directory. Unknown users fall back to the least-privileged role.
_USER_ROLES: dict[str, str] = {}
_DEFAULT_ROLE = "customer"


def get_role(user_id: str) -> Role:
    role_name = _USER_ROLES.get(user_id, _DEFAULT_ROLE)
    return ROLES[role_name]


def filter_domains(role: Role, domains: list[str]) -> list[str]:
    return [domain for domain in domains if domain in role.allowed_domains]


def filter_tools(role: Role, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if role.allowed_tools is None:
        return tools
    return [tool for tool in tools if tool["name"] in role.allowed_tools]
