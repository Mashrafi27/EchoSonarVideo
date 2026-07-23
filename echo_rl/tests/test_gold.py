from echo_rl.data.gold import load_metric, load_all


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("study_id,label,study designation,from_measurement,from_text,text,text_fields\n" + rows)
    return str(p)


def test_load_metric(tmp_path):
    p = _write(tmp_path, "ejection_fraction_regression.csv",
               "st-1,51.0,TRAIN,True,True,txt,conclusions\nst-2,60.0,TEST,True,False,,\n")
    d = load_metric(p)
    assert d["st-1"]["label"] == "51.0"
    assert d["st-2"]["designation"] == "TEST"


def test_load_all(tmp_path):
    _write(tmp_path, "ejection_fraction_regression.csv", "st-1,51.0,TRAIN,True,True,t,c\n")
    _write(tmp_path, "heart_failure_classification.csv", "st-1,1,TRAIN,True,True,t,c\n")
    alld = load_all(str(tmp_path))
    assert alld["st-1"]["ejection_fraction_regression"] == "51.0"
    assert alld["st-1"]["heart_failure_classification"] == "1"
    assert alld["st-1"]["designation"] == "TRAIN"
