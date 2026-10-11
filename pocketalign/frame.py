QUERY_FRAME = "### Query\n{q}\n\n### Reply\n"
STOP_MARK = "\n### Query"


def frame_query(q):
    return QUERY_FRAME.format(q=q.strip())
