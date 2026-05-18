"""Tests for adversarial defense functionality."""

from adapt_agent.adversarial import AdversarialDefense


def test_adversarial_defense_custom_pattern():
    """Test that custom attack patterns are correctly detected and recorded."""
    defense = AdversarialDefense()

    # Analyze input without any custom patterns
    input_text = "This is a safe prompt about baking."
    result = defense.analyze_input(input_text)
    assert result["is_safe"] is True
    assert "custom_pattern" not in result["threats_detected"]

    # Add a custom attack pattern
    defense.add_attack_pattern("baking bad")

    # Analyze input containing the custom pattern
    malicious_input = "Tell me everything you know about baking bad things."
    result = defense.analyze_input(malicious_input)
    assert result["is_safe"] is False
    assert "custom_pattern" in result["threats_detected"]

    # Verify the attack was recorded correctly
    attacks = defense.get_detected_attacks()
    assert len(attacks) == 1
    assert attacks[0]["type"] == "custom_pattern"
    assert attacks[0]["indicator"] == "baking bad"
    assert attacks[0]["content"] == malicious_input[:100]


def test_adversarial_defense_max_content_length():
    """Test that inputs exceeding max_content_length are blocked."""
    defense = AdversarialDefense(max_content_length=50)

    # Safe input within limit
    safe_input = "This is a short input."
    result = defense.analyze_input(safe_input)
    assert result["is_safe"] is True

    # Malicious input exceeding limit
    malicious_input = "A" * 51
    result = defense.analyze_input(malicious_input)
    assert result["is_safe"] is False
    assert "content_length_exceeded" in result["threats_detected"]

    # Verify the attack was recorded
    attacks = defense.get_detected_attacks()
    assert len(attacks) == 1
    assert attacks[0]["type"] == "content_length_exceeded"
