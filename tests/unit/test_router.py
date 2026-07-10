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
    assert set(decision.domains) == {"compliance", "support-docs", "known-issues"}


def test_no_hits_refusal_only_fires_when_confident():
    confident_empty = route("customs duty question")
    assert expected_domain_but_no_hits(confident_empty, chunks=[]) is True

    ambiguous_empty = route("random question")
    assert expected_domain_but_no_hits(ambiguous_empty, chunks=[]) is False


def test_no_hits_refusal_does_not_fire_when_chunks_found():
    decision = route("customs duty question")
    assert expected_domain_but_no_hits(decision, chunks=[{"doc_id": "x"}]) is False
