# Medicine Price Analysis – Localhost Website

This version runs as a normal Flask website (not Streamlit). It reads CSV/Excel files directly from the project's `data` folders. There is no browser upload step. Clients can search by medicine name only; categories remain visible in the analysis sections, not as a search filter.

## Run in VS Code (Windows PowerShell)

Open this project folder in VS Code, then run:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe app.py
```

Open this address in your browser:

http://127.0.0.1:5000

If `python` is not recognized, use the full path to your installed Python executable in place of `python`.

## Where to store data

- Jan Aushadhi files: `data/raw/janaushadhi/`
- HubScrape files for Phase 2: `data/raw/hubscrape/`

The current Jan Aushadhi catalogue is already placed in:

`data/raw/janaushadhi/jan_aushadhi_catalogue.csv`

To update the catalogue later, replace that CSV in the folder. You do not need to change the code or upload it through the website. Restart/refresh the site to read the updated data.

The Jan Aushadhi format supports `Generic Name`, `Drug Code`, `Unit Size`, `MRP`, `Group Name`, and `Sr No`.

## Project structure

```text
Medicine_Price_Analysis_Localhost/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   └── processor.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── data/
    └── raw/
        ├── janaushadhi/
        │   └── jan_aushadhi_catalogue.csv
        └── hubscrape/
            └── tata1mg.csv
```

For Phase 2, place pharmacy export CSV/Excel files directly inside `data/raw/hubscrape/`. The included `tata1mg.csv` is a small sample export (20 records), not a complete medicine catalogue. The dashboard reads it automatically; no upload button is required.

The app is for academic analysis only. Catalogue prices do not indicate current stock or availability and the app does not provide medical advice.
