# Project rules

## README maintenance

After any of the following changes, check whether `README.md` needs updating before closing the task:

- **New dependencies** added or removed (`pyproject.toml` / `uv.lock`) → update the stack section and any install instructions
- **New script or entry point** created → document it under "Utilisation"
- **Analysis methodology changed** (conception offset, moon phase logic, thresholds, confidence interval method) → update the "Méthodologie" section and the summary result in the intro
- **Data source or preprocessing changed** (new filters, new columns, different CSV) → update the "Données" section
- **New Streamlit page or widget** that changes how the app is used → update the "Interface web interactive" description
- **Prerequisite or Python version changed** → update "Prérequis" and "Installation"

If none of the above apply, no README update is needed. When an update is needed, commit it in the same PR/commit as the change that triggered it.
