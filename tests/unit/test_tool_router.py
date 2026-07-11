from app.mcp_clients import tool_router

_ALL_TOOLS = [
    {"name": "getShipmentStatus"},
    {"name": "listShipments"},
    {"name": "updateShipmentEta"},
    {"name": "getAsnStatus"},
    {"name": "listAsns"},
    {"name": "receiveAsn"},
    {"name": "getSalesOrderStatus"},
    {"name": "listSalesOrders"},
    {"name": "allocateInventory"},
    {"name": "createPickList"},
    {"name": "getStockLevel"},
    {"name": "listLowStock"},
    {"name": "adjustStock"},
    {"name": "getReturnStatus"},
    {"name": "listReturns"},
    {"name": "approveReturn"},
]


def test_id_prefix_routes_confidently_to_its_group():
    decision = tool_router.route("What's the status of return RMA4001?")
    assert decision.confident
    assert decision.groups == frozenset({"returns"})


def test_keyword_routes_without_an_id():
    decision = tool_router.route("What's currently low on stock?")
    assert decision.confident
    assert "inventory" in decision.groups


def test_ambiguous_question_offers_no_groups():
    # Offering everything on an ambiguous question let the model fabricate
    # a tracking ID for a plain "hi" and present the made-up result as
    # real — confirmed live. No tools is the safe fallback here, unlike
    # the RAG router's "search everything" (see module docstring).
    decision = tool_router.route("hi")
    assert not decision.confident
    assert decision.groups == frozenset()


def test_filter_tools_narrows_to_matched_group_only():
    decision = tool_router.route("What's the status of return RMA4001?")
    filtered = tool_router.filter_tools(decision, _ALL_TOOLS)
    names = {t["name"] for t in filtered}
    assert names == {"getReturnStatus", "listReturns", "approveReturn"}


def test_filter_tools_ambiguous_question_drops_every_known_tool():
    decision = tool_router.route("hi")
    filtered = tool_router.filter_tools(decision, _ALL_TOOLS)
    assert filtered == []


def test_filter_tools_never_hides_an_unclassified_tool():
    decision = tool_router.route("What's the status of return RMA4001?")
    tools = [*_ALL_TOOLS, {"name": "someBrandNewTool"}]
    filtered = tool_router.filter_tools(decision, tools)
    assert "someBrandNewTool" in {t["name"] for t in filtered}


def test_multiple_ids_route_to_multiple_groups():
    decision = tool_router.route("Compare shipment TRK1001 with return RMA4001")
    assert decision.groups == frozenset({"shipments", "returns"})
