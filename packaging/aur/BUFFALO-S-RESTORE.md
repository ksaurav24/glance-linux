# Reassemble buffalo_s.zip

GitHub rejects individual files over 100 MB. The original `buffalo_s.zip` was split into smaller parts for storage in this repository. From this directory, reassemble it with:

```bash
cat buffalo_s.zip.part-* > buffalo_s.zip
```

The parts are ordered lexically (`aa`, `ab`, ...).
