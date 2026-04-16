def test_clean_column_name_brackets():
    from steel_flow.data import clean_column_name
    assert clean_column_name("time[s]") == "time_s_"


def test_clean_column_name_slash():
    from steel_flow.data import clean_column_name
    assert clean_column_name("AN_1_LL[m/s]") == "AN_1_LL_m_s_"


def test_clean_column_name_already_clean():
    from steel_flow.data import clean_column_name
    assert clean_column_name("already_clean") == "already_clean"


def test_clean_column_name_spaces():
    from steel_flow.data import clean_column_name
    assert clean_column_name("has space") == "has_space"
