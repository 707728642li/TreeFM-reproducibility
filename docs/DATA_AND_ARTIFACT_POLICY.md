# Data and artifact policy

## Not stored in Git

- public or controlled raw sequencing reads;
- genome FASTA, annotation GFF/GTF, and proteome FASTA files;
- pretrained or adapted model weights;
- embeddings, feature matrices, checkpoints, and optimizer states;
- output metrics, paper tables, rendered figures, or narrative result reports;
- manuscripts, supplements, cover letters, reviews, or author metadata.

## Expected external inputs

Inputs must be obtained from the accession and source records described by the dated protocols. Users should retain source URLs, accession identifiers, file sizes, checksums, download dates, and any transformations in a local data manifest outside Git.

## Local layout

Use a writable analysis root supplied through `TREEFM_ROOT`. A typical untracked layout is:

```text
$TREEFM_ROOT/
  data/raw/
  data/processed/
  models/
  embeddings/
  results/
  logs/
```

`PUBLIC_GENOME_ROOT` may point to a read-only institutional mirror. Copy required inputs into the writable project root before processing; never edit a shared public-genome mirror in place.

## Public-release checklist

Before changing the repository from private to public:

1. rerun `tools/validate_repository.py`;
2. search the complete Git history for credentials, author metadata, and absolute paths;
3. confirm third-party code provenance and licenses;
4. obtain author approval for the code license and citation metadata;
5. attach a version tag and immutable archive DOI separately from GitHub;
6. keep large data and trained weights in an accessioned archive rather than Git history.
