from app.rag.domain_sources import DOMAINS
from app.rag.router import expected_domain_but_no_hits, route


def test_compliance_keyword_routes_confidently():
    decision = route("What customs duty applies to this import?")
    assert decision.confident is True
    assert "compliance" in decision.domains


def test_country_tag_routes_to_compliance():
    decision = route("something unrelated", tags={"country": "UAE"})
    assert decision.confident is True
    assert decision.domains == ["compliance"]


def test_support_keywords_route_to_support_and_known_issues():
    decision = route("the app is not working, how do I fix this?")
    assert decision.confident is True
    assert set(decision.domains) == {"support-docs", "known-issues"}


def test_ambiguous_question_falls_back_to_all_domains_unconfident():
    decision = route("What is the status of shipment TRK1001?")
    assert decision.confident is False
    assert set(decision.domains) == set(DOMAINS)


def test_no_hits_refusal_only_fires_when_confident():
    confident_empty = route("customs duty question")
    assert expected_domain_but_no_hits(confident_empty, chunks=[]) is True

    ambiguous_empty = route("random question")
    assert expected_domain_but_no_hits(ambiguous_empty, chunks=[]) is False


def test_no_hits_refusal_does_not_fire_when_chunks_found():
    decision = route("customs duty question")
    assert expected_domain_but_no_hits(decision, chunks=[{"doc_id": "x"}]) is False


def test_inbound_keyword_routes_confidently():
    decision = route("How does ASN receiving work for a new supplier?")
    assert decision.confident is True
    assert decision.domains == ["inbound"]


def test_outbound_keyword_routes_confidently():
    decision = route("What's our backorder policy?")
    assert decision.confident is True
    assert decision.domains == ["outbound"]


def test_inventory_keyword_routes_confidently():
    decision = route("What's the cycle count process?")
    assert decision.confident is True
    assert decision.domains == ["inventory"]


def test_returns_keyword_routes_confidently():
    decision = route("What's the return policy for damaged items?")
    assert decision.confident is True
    assert decision.domains == ["returns"]


def test_live_status_questions_do_not_confidently_match_new_domains():
    # Regression guard: a bare entity noun ("return", "stock") would also
    # match a live tool-lookup question, and a confident RAG match with
    # zero hits short-circuits the whole request into a refusal before
    # the tool-calling step ever runs. New domain keywords must stay
    # specific enough that these stay unconfident (falls back to
    # searching everything, which is safe) rather than confidently
    # matching a documentation domain that has nothing about a specific ID.
    for question in [
        "What's the status of return RMA4001?",
        "What's the status of shipment TRK1001?",
        "What's currently low on stock?",
        "What's the status of sales order SO3001?",
    ]:
        assert route(question).confident is False, question
