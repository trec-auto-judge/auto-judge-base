# Data models

The objects a judge reads: one [`Report`](#autojudge_base.report.Report) per system response, one [`Request`](#autojudge_base.request.Request) per topic, and [`Document`](#autojudge_base.document.document.Document) entries for cited or retrieved text. Because tracks differ in how they attach citations, the sentence types below normalize through `Report.get_sentences_with_citations()`.

## Report

::: autojudge_base.report.Report

## Report metadata and sentences

::: autojudge_base.report.ReportMetaData

::: autojudge_base.report.NeuclirReportSentence

::: autojudge_base.report.RagtimeReportSentence

::: autojudge_base.report.Rag24ReportSentence

::: autojudge_base.report.RankedDocument

::: autojudge_base.report.RetrievedDocuments

## Request

::: autojudge_base.request.Request

## Document

::: autojudge_base.document.document.Document
