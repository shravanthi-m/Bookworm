from bookworm.modes.read_aloud import from_top, strip_running_head


def test_running_heads_are_dropped():
    assert strip_running_head("200 : JOSTEIN GAARDER 'So nature is not a dream") == "'So nature is not a dream"
    assert strip_running_head("SOPHIE'S WORLD : 201 But what we think").startswith("But what")
    assert strip_running_head("I think, therefore I am.") == "I think, therefore I am."


def test_reads_in_order_and_finishes_the_sentence():
    left = "12 : A BOOK One two three. Four five six seven eight nine ten."
    right = "A BOOK : 13 Next page • starts here."
    chunk, rest = from_top([left, right], max_words=5)
    assert chunk == "One two three. Four five six seven eight nine ten."    # ran on to the full stop
    assert rest == "Next page starts here."
    assert from_top([rest], max_words=5) == ("Next page starts here.", "")
