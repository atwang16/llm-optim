def str_seq_to_int(s_traj: str) -> list[list[int]]:
    out = []
    states = s_traj.split(",")
    for state in states:
        state = state.replace(" ", "")  # remove whitespace
        state = [int(s) for s in state]
        out.append(state)
    return out


def int_seq_to_str(states: list[list[int]], delim=" ") -> str:
    return " , ".join([" ".join([str(s) for s in state]) for state in states])
