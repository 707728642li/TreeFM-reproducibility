#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(DESeq2)
  library(BiocParallel)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
project_root <- "."
if (length(args) == 2 && args[[1]] == "--project-root") {
  project_root <- args[[2]]
} else if (length(args) != 0) {
  stop("usage: 128_build_prunus_publication_v3_robust_labels.R [--project-root PATH]")
}
project_root <- normalizePath(project_root, mustWork = TRUE)
contract <- file.path(
  project_root,
  "docs/publication_v3_prunus_functional_label_contract.md"
)
interim_root <- file.path(
  project_root,
  "data/interim/functional_v3/Prunus_publication_v3"
)
output_root <- file.path(
  project_root,
  "results/biological_cases/prunus_publication_v3"
)
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

workers <- suppressWarnings(as.integer(Sys.getenv(
  "TREEFM_DESEQ2_WORKERS",
  unset = "8"
)))
if (is.na(workers) || workers < 1L) {
  workers <- 1L
}
workers <- min(workers, 12L)
bp_param <- if (workers > 1L) {
  MulticoreParam(workers = workers, progressbar = FALSE)
} else {
  SerialParam(progressbar = FALSE)
}

read_study <- function(accession) {
  count_table <- read.delim(
    gzfile(file.path(interim_root, paste0(accession, "_selected_counts.tsv.gz"))),
    header = TRUE,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  if (colnames(count_table)[[1]] != "gene_id") {
    stop(paste(accession, "count table does not begin with gene_id"))
  }
  gene_ids <- count_table$gene_id
  if (anyDuplicated(gene_ids) != 0L) {
    stop(paste(accession, "contains duplicated gene IDs"))
  }
  count_matrix <- as.matrix(count_table[, -1, drop = FALSE])
  storage.mode(count_matrix) <- "integer"
  rownames(count_matrix) <- gene_ids
  metadata <- read.delim(
    file.path(interim_root, paste0(accession, "_selected_design.tsv")),
    header = TRUE,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  if (!setequal(colnames(count_matrix), metadata$sample_id)) {
    stop(paste(accession, "count columns and design samples differ"))
  }
  metadata <- metadata[
    match(colnames(count_matrix), metadata$sample_id),
    ,
    drop = FALSE
  ]
  rownames(metadata) <- metadata$sample_id
  list(
    accession = accession,
    genes = gene_ids,
    counts = count_matrix,
    metadata = metadata
  )
}

random_effect_meta <- function(effects, standard_errors) {
  genes <- nrow(effects)
  result <- matrix(
    NA_real_,
    nrow = genes,
    ncol = 6,
    dimnames = list(
      rownames(effects),
      c(
        "effect", "se", "pvalue", "tau2",
        "direction_fraction", "subgroups"
      )
    )
  )
  for (index in seq_len(genes)) {
    effect <- effects[index, ]
    se <- standard_errors[index, ]
    keep <- is.finite(effect) & is.finite(se) & se > 0
    effect <- effect[keep]
    se <- se[keep]
    k <- length(effect)
    if (k < 2L) {
      next
    }
    fixed_weight <- 1 / se^2
    fixed_effect <- sum(fixed_weight * effect) / sum(fixed_weight)
    q_value <- sum(fixed_weight * (effect - fixed_effect)^2)
    c_value <- sum(fixed_weight) -
      sum(fixed_weight^2) / sum(fixed_weight)
    tau2 <- if (c_value > 0) {
      max(0, (q_value - (k - 1)) / c_value)
    } else {
      0
    }
    random_weight <- 1 / (se^2 + tau2)
    meta_effect <- sum(random_weight * effect) / sum(random_weight)
    meta_se <- sqrt(1 / sum(random_weight))
    z_value <- meta_effect / meta_se
    result[index, ] <- c(
      meta_effect,
      meta_se,
      2 * pnorm(-abs(z_value)),
      tau2,
      mean(sign(effect) == sign(meta_effect)),
      k
    )
  }
  as.data.frame(result, check.names = FALSE)
}

fit_study <- function(study) {
  metadata <- study$metadata
  counts_matrix <- study$counts
  normalization_dds <- DESeqDataSetFromMatrix(
    countData = counts_matrix,
    colData = metadata,
    design = ~ 1
  )
  normalization_dds <- estimateSizeFactors(normalization_dds)
  normalized <- counts(normalization_dds, normalized = TRUE)
  detectable_count <- rowSums(normalized >= 10)
  detectable <- detectable_count >= ceiling(ncol(normalized) / 2)

  subgroup_names <- sort(unique(metadata$subgroup))
  effects <- matrix(
    NA_real_,
    nrow = nrow(counts_matrix),
    ncol = length(subgroup_names),
    dimnames = list(rownames(counts_matrix), subgroup_names)
  )
  standard_errors <- effects
  raw_pvalues <- effects
  for (subgroup in subgroup_names) {
    samples <- rownames(metadata)[metadata$subgroup == subgroup]
    subgroup_metadata <- metadata[samples, , drop = FALSE]
    subgroup_metadata$condition <- factor(
      subgroup_metadata$condition,
      levels = c("baseline", "endpoint")
    )
    cell_counts <- table(subgroup_metadata$condition)
    if (length(cell_counts) != 2L || any(cell_counts < 2L)) {
      stop(paste(study$accession, subgroup, "has an invalid two-cell design"))
    }
    dds <- DESeqDataSetFromMatrix(
      countData = counts_matrix[, samples, drop = FALSE],
      colData = subgroup_metadata,
      design = ~ condition
    )
    dds <- DESeq(
      dds,
      test = "Wald",
      parallel = workers > 1L,
      BPPARAM = bp_param,
      quiet = TRUE
    )
    result <- as.data.frame(results(
      dds,
      contrast = c("condition", "endpoint", "baseline"),
      alpha = 0.10,
      independentFiltering = TRUE,
      parallel = workers > 1L,
      BPPARAM = bp_param
    ))
    effects[, subgroup] <- result$log2FoldChange
    standard_errors[, subgroup] <- result$lfcSE
    raw_pvalues[, subgroup] <- result$pvalue
  }
  meta <- random_effect_meta(effects, standard_errors)
  meta$padj <- p.adjust(meta$pvalue, method = "BH")
  z90 <- qnorm(0.95)
  meta$ci90_lower <- meta$effect - z90 * meta$se
  meta$ci90_upper <- meta$effect + z90 * meta$se
  meta$detectable_count <- detectable_count
  meta$detectable <- detectable
  list(
    accession = study$accession,
    meta = meta,
    effects = effects,
    standard_errors = standard_errors,
    raw_pvalues = raw_pvalues,
    samples = ncol(counts_matrix),
    subgroups = subgroup_names,
    size_factors = sizeFactors(normalization_dds)
  )
}

accessions <- c("GSE130426", "GSE138792", "GSE298924")
studies <- lapply(accessions, read_study)
gene_ids <- studies[[1]]$genes
if (!all(vapply(studies, function(item) identical(item$genes, gene_ids), logical(1)))) {
  stop("Prunus study gene orders differ")
}
fits <- lapply(studies, fit_study)
names(fits) <- accessions

meta_effect <- do.call(
  cbind,
  lapply(fits, function(item) item$meta$effect)
)
meta_se <- do.call(cbind, lapply(fits, function(item) item$meta$se))
meta_padj <- do.call(cbind, lapply(fits, function(item) item$meta$padj))
meta_ci90_lower <- do.call(
  cbind,
  lapply(fits, function(item) item$meta$ci90_lower)
)
meta_ci90_upper <- do.call(
  cbind,
  lapply(fits, function(item) item$meta$ci90_upper)
)
direction_fraction <- do.call(
  cbind,
  lapply(fits, function(item) item$meta$direction_fraction)
)
detectable <- do.call(
  cbind,
  lapply(fits, function(item) item$meta$detectable)
)
colnames(meta_effect) <- accessions
colnames(meta_se) <- accessions
colnames(meta_padj) <- accessions
colnames(meta_ci90_lower) <- accessions
colnames(meta_ci90_upper) <- accessions
colnames(direction_fraction) <- accessions
colnames(detectable) <- accessions

finite_nonzero <- is.finite(meta_effect) & sign(meta_effect) != 0
same_direction <- (
  rowSums(finite_nonzero) == length(accessions) &
    apply(sign(meta_effect), 1, function(value) length(unique(value)) == 1L)
)
study_positive_support <- (
  !is.na(meta_padj) &
    meta_padj <= 0.10 &
    abs(meta_effect) >= 0.75 &
    direction_fraction >= (2 / 3)
)
median_absolute_effect <- apply(abs(meta_effect), 1, median, na.rm = TRUE)
positive <- (
  rowSums(detectable) == length(accessions) &
    same_direction &
    rowSums(study_positive_support) >= 2L &
    median_absolute_effect >= 0.75
)

study_equivalent <- (
  is.finite(meta_ci90_lower) &
    is.finite(meta_ci90_upper) &
    meta_ci90_lower >= -0.75 &
    meta_ci90_upper <= 0.75
)
no_study_significant <- rowSums(
  !is.na(meta_padj) & meta_padj < 0.10
) == 0L
all_study_effects_below_one <- (
  rowSums(is.finite(meta_effect) & abs(meta_effect) < 1.0) ==
    length(accessions)
)
negative <- (
  rowSums(detectable) == length(accessions) &
    no_study_significant &
    all_study_effects_below_one &
    rowSums(study_equivalent) >= 2L
)
if (any(positive & negative)) {
  stop("Prunus positive and negative definitions overlap")
}

label <- rep("ambiguous", length(gene_ids))
label[negative] <- "negative"
label[positive] <- "positive"
endpoint_direction <- rep("none", length(gene_ids))
endpoint_direction[positive & meta_effect[, 1] > 0] <- "up"
endpoint_direction[positive & meta_effect[, 1] < 0] <- "down"

result <- data.frame(
  gene_id = gene_ids,
  detectable_all_studies = rowSums(detectable) == length(accessions),
  same_study_meta_direction = same_direction,
  positive_supporting_studies = rowSums(study_positive_support),
  equivalent_studies = rowSums(study_equivalent),
  median_absolute_study_effect = median_absolute_effect,
  no_study_padj_below_point_one = no_study_significant,
  all_study_effects_below_one = all_study_effects_below_one,
  endpoint_direction = endpoint_direction,
  label = label,
  check.names = FALSE
)
for (accession in accessions) {
  fit <- fits[[accession]]
  result[[paste0(accession, "_meta_lfc")]] <- fit$meta$effect
  result[[paste0(accession, "_meta_se")]] <- fit$meta$se
  result[[paste0(accession, "_meta_pvalue")]] <- fit$meta$pvalue
  result[[paste0(accession, "_meta_padj")]] <- fit$meta$padj
  result[[paste0(accession, "_meta_ci90_lower")]] <- fit$meta$ci90_lower
  result[[paste0(accession, "_meta_ci90_upper")]] <- fit$meta$ci90_upper
  result[[paste0(accession, "_direction_fraction")]] <-
    fit$meta$direction_fraction
  result[[paste0(accession, "_detectable_count")]] <-
    fit$meta$detectable_count
  result[[paste0(accession, "_detectable")]] <- fit$meta$detectable
  for (subgroup in fit$subgroups) {
    safe_name <- gsub("[^A-Za-z0-9]+", "_", subgroup)
    result[[paste0(accession, "_", safe_name, "_lfc")]] <-
      fit$effects[, subgroup]
    result[[paste0(accession, "_", safe_name, "_se")]] <-
      fit$standard_errors[, subgroup]
  }
}

label_path <- file.path(
  output_root,
  "prunus_publication_v3_robust_labels.tsv.gz"
)
connection <- gzfile(label_path, open = "wt", compression = 6)
write.table(
  result,
  file = connection,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = "NA"
)
close(connection)
saveRDS(
  list(
    fits = fits,
    accessions = accessions,
    contract = contract
  ),
  file.path(output_root, "prunus_publication_v3_models.rds"),
  compress = "xz"
)
writeLines(
  capture.output(sessionInfo()),
  file.path(output_root, "session_info.txt")
)

positive_count <- sum(positive)
negative_count <- sum(negative)
gate_pass <- positive_count >= 200L && negative_count >= 500L
summary <- list(
  status = "Prunus_publication_v3_robust_labels_built",
  contract = "docs/publication_v3_prunus_functional_label_contract.md",
  deseq2_version = as.character(packageVersion("DESeq2")),
  workers = workers,
  common_genes = length(gene_ids),
  studies = lapply(accessions, function(accession) {
    fit <- fits[[accession]]
    list(
      accession = accession,
      samples = fit$samples,
      subgroups = as.list(fit$subgroups),
      size_factor_min = min(fit$size_factors),
      size_factor_median = median(fit$size_factors),
      size_factor_max = max(fit$size_factors),
      detectable_genes = sum(fit$meta$detectable)
    )
  }),
  detectable_all_studies = sum(rowSums(detectable) == length(accessions)),
  same_direction_genes = sum(same_direction),
  positive_genes = positive_count,
  positive_up = sum(positive & endpoint_direction == "up"),
  positive_down = sum(positive & endpoint_direction == "down"),
  negative_genes = negative_count,
  ambiguous_genes = sum(label == "ambiguous"),
  minimum_positive_genes = 200L,
  minimum_negative_genes = 500L,
  label_gate_pass = gate_pass,
  output_labels = basename(label_path),
  model_objects = "prunus_publication_v3_models.rds",
  session_info = "session_info.txt"
)
write_json(
  summary,
  file.path(output_root, "label_gate_summary.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = 10
)
cat(toJSON(summary, auto_unbox = TRUE), "\n")

