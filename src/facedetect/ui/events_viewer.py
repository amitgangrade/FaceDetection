from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

from ..storage.repo import PersonRepo


class EventsViewer(tk.Toplevel):
    def __init__(self, parent: tk.Misc, repo: PersonRepo):
        super().__init__(parent)
        self.title("Recent events")
        self.geometry("720x420")
        self.repo = repo
        self._photo_ref = None
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        cols = ("when", "who", "similarity", "snapshot")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=16)
        for c, w in zip(cols, (150, 160, 80, 220)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.config(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        right = ttk.Frame(main, padding=(8, 0, 0, 0))
        right.pack(side="left", fill="y")
        self.thumb_label = ttk.Label(right)
        self.thumb_label.pack()

        btn_frame = ttk.Frame(right)
        btn_frame.pack(pady=8, fill="x")
        ttk.Button(btn_frame, text="Refresh", command=self._refresh).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Close", command=self.destroy).pack(fill="x", pady=2)

    def _refresh(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._events = self.repo.recent_events(limit=100)
        people = {p.id: p.name for p in self.repo.list_people()}
        for e in self._events:
            who = people.get(e.person_id, "unknown") if e.person_id else "unknown"
            sim = f"{e.similarity:.2f}" if e.similarity is not None else "-"
            snap = e.snapshot_path or ""
            self.tree.insert("", "end", iid=str(e.id), values=(e.frame_ts, who, sim, snap))

    def _on_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        event_id = int(sel[0])
        event = next((e for e in self._events if e.id == event_id), None)
        if event is None or not event.snapshot_path:
            self.thumb_label.configure(image="")
            self._photo_ref = None
            return
        p = Path(event.snapshot_path)
        if not p.exists():
            self.thumb_label.configure(image="")
            return
        try:
            img = Image.open(p)
            img.thumbnail((260, 260))
            photo = ImageTk.PhotoImage(img)
            self._photo_ref = photo
            self.thumb_label.configure(image=photo)
        except Exception:  # noqa: BLE001
            self.thumb_label.configure(image="")
