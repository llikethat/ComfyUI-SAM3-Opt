# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Stub module for inference-only usage
# The matcher is only used during training, not inference

class BinaryHungarianMatcherV2:
    """Stub matcher class - only used during training, not inference."""
    
    def __init__(self, **kwargs):
        raise NotImplementedError(
            "BinaryHungarianMatcherV2 is only available during training. "
            "For inference, use eval_mode=True."
        )

__all__ = ["BinaryHungarianMatcherV2"]
