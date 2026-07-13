# Qrels

A judge's `create_qrels()` returns [`Qrels`](#autojudge_base.qrels.Qrels) — `(topic_id, doc_id, grade)` rows. Define extractor functions in a `QrelsSpec`, build with `build_qrels`, verify with `Qrels.verify(...)`, and serialize to TREC format with `write_qrel_file`. For generated text without a corpus id, derive a stable id with `doc_id_md5`.

## Qrels and rows

::: autojudge_base.qrels.Qrels

::: autojudge_base.qrels.QrelRow

## Spec and builder

::: autojudge_base.qrels.QrelsSpec

::: autojudge_base.qrels.build_qrels

## Verification and I/O

::: autojudge_base.qrels.QrelsVerification

::: autojudge_base.qrels.QrelsVerificationError

::: autojudge_base.qrels.write_qrel_file

::: autojudge_base.qrels.doc_id_md5
