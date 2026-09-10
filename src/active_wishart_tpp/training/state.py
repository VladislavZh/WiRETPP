"""Detached neural snapshots for common and selected checkpoints."""


def clone_state_dict(model):
    """Take a detached CPU snapshot of the neural bank."""
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
