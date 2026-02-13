def evaluate_drop_decision(hit_probability, user_threshold):
    """
    DROP if hit_probability >= user_threshold, else NO DROP.
    Deterministic, no side effects.
    """
    if hit_probability >= user_threshold:
        return "DROP"
    return "NO DROP"
