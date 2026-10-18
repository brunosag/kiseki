// One report as a document: the models it compares, then a section per analysis with its
// tables and figures, and an outline to jump between them. While it builds, the analyses
// tick off as they finish.
import { CheckIcon, TriangleAlertIcon } from "lucide-react";
import { Fragment } from "react";
import { Crumbs, PageBody, PageHeader } from "@/components/page";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Dialog, DialogPopup, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Spinner } from "@/components/ui/spinner";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { fileURL } from "@/lib/api";
import { analysisTitle, dateTime, datasetTitle, duration, idTime, optimizerTitle, shortId, words } from "@/lib/format";
import { useLive } from "@/lib/live";
import { blocks, inline } from "@/lib/markdown";
import { refParts, reportTitle, runOf } from "@/lib/reports";
import { useResource } from "@/lib/resource";
import { href } from "@/lib/router";
import type { Options, Report, ReportDetail, ReportDocument, Section, Subject } from "@/lib/types";
import { cn } from "@/lib/utils";

export function ReportView({ id }: { id: string }) {
  const { reports, runs, options } = useLive();
  const listed = reports?.find((report) => report.id === id);
  const detail = useResource<ReportDetail>(`reports/${encodeURIComponent(id)}`, listed?.status.state);
  const report: Report | undefined = listed ?? detail.data;
  const title = report ? reportTitle(report, runs, options) : shortId(id);
  const document = detail.data?.report ?? null;
  const dataset = document?.dataset ?? runOf(report?.spec.models[0] ?? "", runs)?.spec.dataset;

  return (
    <>
      <PageHeader>
        <Crumbs items={[{ label: "Reports", href: "/reports" }, { label: title }]} />
      </PageHeader>
      {!report ? (
        <Empty className="flex-1">
          {detail.error ? (
            <EmptyHeader>
              <EmptyTitle className="font-medium text-base">No such report</EmptyTitle>
              <EmptyDescription>{detail.error}</EmptyDescription>
            </EmptyHeader>
          ) : (
            <Spinner />
          )}
        </Empty>
      ) : (
        <PageBody className="scroll-pt-6">
          <div className="mx-auto flex max-w-6xl justify-center gap-12 px-8 py-7">
            <article className="min-w-0 max-w-3xl flex-1">
              <h1 className="font-medium text-xl tracking-tight">{title}</h1>
              <p className="mt-1 text-muted-foreground text-xs">
                {[
                  dataset && datasetTitle(options, dataset),
                  report.spec.split === "val" ? "validation split" : "test split",
                  `seed ${report.spec.seed}`,
                  dateTime(idTime(report.id)),
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
              {report.status.state === "failed" && (
                <Alert variant="error" className="mt-5">
                  <TriangleAlertIcon />
                  <AlertTitle>
                    Failed
                    {report.status.stage &&
                      ` in ${analysisTitle(options, report.status.stage)}`}
                  </AlertTitle>
                  <AlertDescription className="font-mono text-xs">{report.status.error}</AlertDescription>
                </Alert>
              )}
              {document ? (
                <>
                  <Subjects subjects={document.subjects} options={options} />
                  {document.sections.map((section) => (
                    <SectionView key={section.name} id={report.id} section={section} document={document} />
                  ))}
                </>
              ) : (
                <Progress report={report} options={options} />
              )}
            </article>
            {document && <Outline sections={document.sections} />}
          </div>
        </PageBody>
      )}
    </>
  );
}

function Subjects({ subjects, options }: { subjects: Subject[]; options: Options | null }) {
  return (
    <ul className="mt-6 divide-y rounded-xl border">
      {subjects.map((subject) => {
        const { run, model } = refParts(subject.ref);
        return (
          <li key={subject.ref} className="flex h-10 items-center gap-3 px-4 text-[13px]">
            <span className="size-2 shrink-0 rounded-full" style={{ background: subject.color }} />
            <a href={href.run(run)} className="font-medium hover:underline">
              {subject.label}
            </a>
            <span className="text-muted-foreground">
              {optimizerTitle(options, subject.optimizer)} · <span className="font-mono text-xs">{subject.model}</span>
            </span>
            <span className="ms-auto text-muted-foreground text-xs">{model === "best" ? "best model" : model}</span>
          </li>
        );
      })}
    </ul>
  );
}

/** The analyses, done, in progress and waiting, while the report builds. */
function Progress({ report, options }: { report: Report; options: Options | null }) {
  const { status } = report;
  const seconds = status.seconds ?? {};
  return (
    <ol className="mt-6 divide-y rounded-xl border">
      {Object.keys(report.spec.analyses).map((name) => {
        const done = name in seconds;
        const current = status.state === "running" && status.stage === name;
        const failed = status.state === "failed" && status.stage === name;
        return (
          <li key={name} className="flex h-10 items-center gap-3 px-4 text-[13px]">
            <span className="flex size-4 items-center justify-center">
              {done ? (
                <CheckIcon className="size-3.5 text-indigo-500 dark:text-indigo-400" />
              ) : current ? (
                <Spinner className="size-3.5" />
              ) : failed ? (
                <TriangleAlertIcon className="size-3.5 text-destructive-foreground" />
              ) : (
                <span className="size-1.5 rounded-full bg-muted-foreground/40" />
              )}
            </span>
            <span className={cn(!done && !current && "text-muted-foreground")}>
              {analysisTitle(options, name)}
            </span>
            <span className="ms-auto text-muted-foreground text-xs tabular-nums">
              {done ? duration(seconds[name]) : current ? "running" : ""}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function SectionView({ id, section, document }: { id: string; section: Section; document: ReportDocument }) {
  const colors = new Map(document.subjects.map((subject) => [subject.label, subject.color]));
  return (
    <section id={section.name} className="mt-12">
      <h2 className="mb-4 font-medium text-base tracking-tight">{section.title}</h2>
      <Markdown text={section.markdown} colors={colors} />
      {section.figures.map((figure) => (
        <Figure
          key={figure}
          url={fileURL("reports", id, figure)}
          alt={`${section.title}: ${words(figure.replace(/^figures\/|\.png$/g, ""))}`}
        />
      ))}
    </section>
  );
}

/** A figure on its white card; a click shows it full size above the report. */
function Figure({ url, alt }: { url: string; alt: string }) {
  return (
    <Dialog>
      <DialogTrigger className="mt-5 block w-full cursor-zoom-in overflow-hidden rounded-xl border bg-white p-3" title="Full size">
        <img src={url} alt={alt} className="mx-auto" loading="lazy" />
      </DialogTrigger>
      <DialogPopup
        className="w-auto max-w-[calc(100vw-4rem)] overflow-auto bg-white p-4 sm:max-w-[calc(100vw-4rem)]"
        bottomStickOnMobile={false}
        closeProps={{ className: "absolute end-2 top-2 text-neutral-500 hover:bg-black/5 hover:text-neutral-900" }}
      >
        <DialogTitle className="sr-only">{alt}</DialogTitle>
        <img src={url} alt={alt} className="max-w-none" />
      </DialogPopup>
    </Dialog>
  );
}

const NUMERIC = /^[-+−]?[\d.,]+(e[-+]?\d+)?\s*%?$/i;

/** A section's text: paragraphs and tables. A cell naming a model gets that model's color. */
function Markdown({ text, colors }: { text: string; colors: Map<string, string> }) {
  return blocks(text).map((block, index) =>
    block.kind === "paragraph" ? (
      <p key={index} className="my-3 max-w-prose text-[13px] text-muted-foreground leading-relaxed">
        <Inline text={block.text} />
      </p>
    ) : (
      <div key={index} className="my-4">
        <Table className="text-[13px] tabular-nums">
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              {block.head.map((cell, column) => (
                <TableHead
                  key={column}
                  className={cn(
                    "h-8 px-3 font-normal text-xs first:ps-0 last:pe-0",
                    column > 0 && block.rows.every((row) => NUMERIC.test(row[column] ?? "")) ? "text-right" : "text-left",
                  )}
                >
                  <Inline text={cell} />
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {block.rows.map((row, line) => (
              <TableRow key={line}>
                {row.map((cell, column) => (
                  <TableCell
                    key={column}
                    className={cn(
                      "h-8 px-3 py-0 first:ps-0 last:pe-0",
                      column > 0 && NUMERIC.test(cell) && "text-right",
                    )}
                  >
                    {colors.has(cell) ? (
                      <span className="inline-flex items-center gap-2">
                        <span className="size-2 rounded-full" style={{ background: colors.get(cell) }} />
                        {cell}
                      </span>
                    ) : (
                      <Inline text={cell} />
                    )}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    ),
  );
}

function Inline({ text }: { text: string }) {
  return inline(text).map((span, index) => (
    <Fragment key={index}>
      {typeof span === "string" ? (
        span
      ) : "bold" in span ? (
        <strong className="font-medium text-foreground">{span.bold}</strong>
      ) : (
        <code className="rounded bg-muted px-1 text-xs">{span.code}</code>
      )}
    </Fragment>
  ));
}

function Outline({ sections }: { sections: Section[] }) {
  return (
    <nav aria-label="On this page" className="sticky top-7 hidden w-52 shrink-0 self-start xl:block">
      <h3 className="mb-2 font-medium text-muted-foreground text-xs">On this page</h3>
      <ul className="space-y-0.5 border-s">
        {sections.map((section) => (
          <li key={section.name}>
            <a
              href={`#${section.name}`}
              onClick={(event) => {
                event.preventDefault();
                window.document.getElementById(section.name)?.scrollIntoView({ behavior: "smooth" });
              }}
              className="-ms-px block border-transparent border-s py-1 ps-3 text-[13px] text-muted-foreground transition-colors hover:border-foreground/40 hover:text-foreground"
            >
              {section.title}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
