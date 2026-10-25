"""Optional third-party integrations for RoadWatch.

Rules enforced by this package:
- Everything here is OPTIONAL and OFF by default. The core pipeline, models and dashboard never import it unconditionally.
- No remote call is made at import or application start-up; remote calls happen only after an explicit user action.
- Nothing here can change a model score, a ranking or a worklist, or issue a repair order. Integrations only read
  saved, verified outputs and present or anchor them.
- Credentials come from the environment (or the git-ignored .env) and are never logged, cached or written to outputs.
"""
