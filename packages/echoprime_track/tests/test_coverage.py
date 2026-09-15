from echoprime_track.coverage import missing_dicom_uuids


def test_missing_dicom_uuids_reports_gap():
    needed = {"di-0001", "di-0002", "di-0003"}
    available = {"di-0001", "di-0003"}
    assert missing_dicom_uuids(needed, available) == {"di-0002"}


def test_missing_dicom_uuids_full_coverage():
    needed = {"di-0001", "di-0002"}
    available = {"di-0001", "di-0002", "di-0099"}
    assert missing_dicom_uuids(needed, available) == set()
