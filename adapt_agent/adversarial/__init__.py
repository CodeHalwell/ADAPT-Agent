"""Adversarial defense for LLM agents."""

from typing import Any, Dict, List, Optional
from datetime import datetime


class AdversarialDefense:
    """Defends against adversarial attacks on LLM agents.
    
    Provides detection and mitigation strategies for common attack vectors
    including prompt injection, jailbreaking, and data poisoning.
    """
    
    def __init__(self):
        """Initialize the AdversarialDefense."""
        self._attack_patterns: List[str] = []
        self._detected_attacks: List[Dict[str, Any]] = []
        self._defense_strategies: Dict[str, Any] = {}
    
    def detect_prompt_injection(self, prompt: str) -> bool:
        """Detect potential prompt injection attacks.
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            True if attack detected, False otherwise
        """
        # Common prompt injection patterns
        injection_indicators = [
            "ignore previous instructions",
            "disregard all",
            "new instructions:",
            "system:",
            "override",
        ]
        
        prompt_lower = prompt.lower()
        for indicator in injection_indicators:
            if indicator in prompt_lower:
                self._record_attack("prompt_injection", prompt, indicator)
                return True
        
        return False
    
    def detect_jailbreak(self, prompt: str) -> bool:
        """Detect jailbreak attempts.
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            True if jailbreak detected, False otherwise
        """
        # Common jailbreak patterns
        jailbreak_indicators = [
            "pretend you are",
            "roleplay as",
            "act as if",
            "you are now",
        ]
        
        prompt_lower = prompt.lower()
        for indicator in jailbreak_indicators:
            if indicator in prompt_lower:
                self._record_attack("jailbreak", prompt, indicator)
                return True
        
        return False
    
    def analyze_input(self, input_text: str) -> Dict[str, Any]:
        """Analyze input for multiple attack vectors.
        
        Args:
            input_text: Input text to analyze
            
        Returns:
            Analysis results with detected threats
        """
        threats = []
        
        if self.detect_prompt_injection(input_text):
            threats.append("prompt_injection")
        
        if self.detect_jailbreak(input_text):
            threats.append("jailbreak")
        
        return {
            "input": input_text[:100],  # Truncated for privacy
            "threats_detected": threats,
            "is_safe": len(threats) == 0,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def add_attack_pattern(self, pattern: str) -> None:
        """Add a custom attack pattern to detect.
        
        Args:
            pattern: Attack pattern string
        """
        self._attack_patterns.append(pattern)
    
    def get_detected_attacks(
        self,
        attack_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get detected attacks.
        
        Args:
            attack_type: Filter by attack type
            limit: Maximum number of attacks to return
            
        Returns:
            List of detected attacks
        """
        attacks = self._detected_attacks
        
        if attack_type:
            attacks = [a for a in attacks if a["type"] == attack_type]
        
        if limit:
            attacks = attacks[-limit:]
        
        return attacks
    
    def _record_attack(
        self,
        attack_type: str,
        content: str,
        indicator: str,
    ) -> None:
        """Record a detected attack.
        
        Args:
            attack_type: Type of attack
            content: Attack content
            indicator: Indicator that triggered detection
        """
        attack = {
            "type": attack_type,
            "content": content[:100],  # Truncated
            "indicator": indicator,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._detected_attacks.append(attack)


__all__ = ["AdversarialDefense"]
