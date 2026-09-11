#!/usr/bin/env bash
# Export the current commit without .git history for transfer to a public repo.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
DEFAULT_DESTINATION="$(dirname "$ROOT")/wildboar-france-code-only"
DESTINATION="${1:-$DEFAULT_DESTINATION}"

if [[ "$DESTINATION" != /* ]]; then
  DESTINATION="$PWD/$DESTINATION"
fi

cd "$ROOT"
python scripts/validate_repository.py

if [[ -e "$DESTINATION" ]]; then
  echo "Destination already exists; choose a new directory: $DESTINATION" >&2
  exit 1
fi
mkdir -p "$DESTINATION"
git archive --format=tar HEAD | tar -xf - -C "$DESTINATION"

python "$DESTINATION/scripts/validate_repository.py" \
  --root "$DESTINATION" \
  --all-files

cat <<EOF

Created a history-free code export at:
  $DESTINATION

The export contains this commit without Git history. To create a separate
repository from it, use a new empty remote. Example commands:

  cd "$DESTINATION"
  git init -b main
  git add .
  git commit -m "Initial public code release"
  # Add the URL of a newly created empty remote, then push.
EOF
