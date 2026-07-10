from app.auth.permissions import ROLES, filter_domains, filter_tools, get_role


def test_unknown_user_defaults_to_least_privileged_role():
    role = get_role("someone-not-in-the-directory")
    assert role.name == "customer"


def test_customer_cannot_see_known_issues():
    role = ROLES["customer"]
    assert "known-issues" not in role.allowed_domains
    assert "compliance" in role.allowed_domains  # customer-facing per SKILL.md


def test_filter_domains_drops_disallowed():
    role = ROLES["customer"]
    filtered = filter_domains(role, ["compliance", "known-issues", "support-docs"])
    assert set(filtered) == {"compliance", "support-docs"}


def test_filter_tools_none_means_all_allowed():
    role = ROLES["manager"]
    tools = [{"name": "getShipmentStatus"}, {"name": "updateShipmentEta"}]
    assert filter_tools(role, tools) == tools


def test_filter_tools_restricts_to_allowed_set():
    role = ROLES["customer"]
    tools = [{"name": "getShipmentStatus"}, {"name": "updateShipmentEta"}]
    filtered = filter_tools(role, tools)
    assert [t["name"] for t in filtered] == ["getShipmentStatus"]
