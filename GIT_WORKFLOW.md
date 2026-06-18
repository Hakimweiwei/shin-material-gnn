# Git Workflow & Conventional Commits

AI, saat membuat pesan `git commit`, Anda DILARANG menggunakan pesan sembarangan seperti "update", "fix bug", atau "wip". Anda WAJIB menggunakan format *Conventional Commits* berikut:

## Format: `<type>: <description>`

### Types yang Diizinkan:
- `feat`: Fitur baru (contoh: `feat: implement custom RDKit featurizer for ESOL dataset`)
- `fix`: Perbaikan bug (contoh: `fix: resolve tensor shape mismatch in GINEConv layer`)
- `docs`: Perubahan dokumentasi/file .md (contoh: `docs: update DATA_PIPELINE_LOGIC with edge attributes`)
- `refactor`: Perubahan struktur kode tanpa mengubah fungsi (contoh: `refactor: modularize training loop into trainer.py`)
- `test`: Penambahan atau perbaikan unit test (contoh: `test: add unit tests for SMILES validation`)
- `chore`: Setup environment, gitignore, requirements (contoh: `chore: add .gitignore for python and PyG cache`)

## Aturan Deskripsi:
- Gunakan bahasa Inggris.
- Gunakan huruf kecil semua (lowercase).
- Maksimal 50 karakter.
- Jelaskan "APA" yang dilakukan, bukan "BAGAIMANA" caranya.