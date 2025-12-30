"""Patches and fixes for LLM agent frameworks."""

from typing import Any, Dict, List, Optional


class PatchManager:
    """Manages patches and fixes for LLM agent frameworks.
    
    Provides a system for applying framework-specific patches
    to address bugs, security issues, or add enhancements.
    """
    
    def __init__(self):
        """Initialize the PatchManager."""
        self._patches: Dict[str, Dict[str, Any]] = {}
        self._applied_patches: List[str] = []
    
    def register_patch(
        self,
        patch_id: str,
        framework: str,
        description: str,
        patch_func: Any,
        version_requirement: Optional[str] = None,
    ) -> None:
        """Register a patch.
        
        Args:
            patch_id: Unique identifier for the patch
            framework: Target framework name
            description: Patch description
            patch_func: Function that applies the patch
            version_requirement: Optional version requirement
        """
        self._patches[patch_id] = {
            "framework": framework,
            "description": description,
            "patch_func": patch_func,
            "version_requirement": version_requirement,
        }
    
    def apply_patch(self, patch_id: str, target: Any) -> bool:
        """Apply a patch to a target.
        
        Args:
            patch_id: Patch identifier
            target: Target object to patch
            
        Returns:
            True if patch applied successfully, False otherwise
        """
        if patch_id not in self._patches:
            return False
        
        if patch_id in self._applied_patches:
            # Already applied
            return True
        
        patch = self._patches[patch_id]
        
        try:
            patch["patch_func"](target)
            self._applied_patches.append(patch_id)
            return True
        except Exception as e:
            print(f"Error applying patch {patch_id}: {e}")
            return False
    
    def list_patches(
        self,
        framework: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List available patches.
        
        Args:
            framework: Filter by framework name
            
        Returns:
            List of patch metadata
        """
        patches = []
        
        for patch_id, patch_data in self._patches.items():
            if framework and patch_data["framework"] != framework:
                continue
            
            patches.append({
                "patch_id": patch_id,
                "framework": patch_data["framework"],
                "description": patch_data["description"],
                "applied": patch_id in self._applied_patches,
            })
        
        return patches
    
    def is_applied(self, patch_id: str) -> bool:
        """Check if a patch has been applied.
        
        Args:
            patch_id: Patch identifier
            
        Returns:
            True if applied, False otherwise
        """
        return patch_id in self._applied_patches


__all__ = ["PatchManager"]
