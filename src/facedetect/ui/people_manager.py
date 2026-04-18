from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from ..storage.repo import PersonRepo


class PeopleManager(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        repo: PersonRepo,
        on_add_samples,
        *,
        is_live_active=None,
    ):
        super().__init__(parent)
        self.title("Manage people")
        self.geometry("520x380")
        self.repo = repo
        self.on_add_samples = on_add_samples  # callback(person_id, name) -> None
        self._is_live_active = is_live_active or (lambda: False)
        self._photo_ref = None

        self._build_ui()
        self._refresh()
        self._sync_add_samples_state()
        self.after(500, self._tick_live_state)

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=8)
        main.pack(fill="both", expand=True)

        list_frame = ttk.Frame(main)
        list_frame.pack(side="left", fill="y")

        self.listbox = tk.Listbox(list_frame, width=28, height=16)
        self.listbox.pack(side="left", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)

        right = ttk.Frame(main, padding=(12, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)

        self.thumb_label = ttk.Label(right)
        self.thumb_label.pack()

        self.info_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.info_var, justify="left", wraplength=240).pack(pady=8, fill="x")

        btn_frame = ttk.Frame(right)
        btn_frame.pack(pady=8, fill="x")
        self._add_btn = ttk.Button(btn_frame, text="Add samples", command=self._add_samples)
        self._add_btn.pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Delete person", command=self._delete).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Close", command=self.destroy).pack(fill="x", pady=(12, 2))

        self._live_hint = ttk.Label(
            right,
            text="",
            foreground="#B00020",
            wraplength=240,
            justify="left",
        )
        self._live_hint.pack(fill="x", pady=(4, 0))

    def _refresh(self) -> None:
        self.listbox.delete(0, "end")
        self._people = self.repo.list_people()
        for p in self._people:
            self.listbox.insert("end", p.name)
        self.info_var.set("")
        self.thumb_label.configure(image="")
        self._photo_ref = None

    def _selected_person(self):
        sel = self.listbox.curselection()
        if not sel:
            return None
        return self._people[sel[0]]

    def _on_select(self, _event=None) -> None:
        p = self._selected_person()
        if p is None:
            return
        n_embeds = self.repo.count_embeddings(p.id)
        events = self.repo.events_for_person(p.id, limit=1)
        last_seen = events[0].frame_ts if events else "never"
        info = (
            f"Name: {p.name}\n"
            f"ID: {p.id}\n"
            f"Enrolled: {p.created_at}\n"
            f"Samples: {n_embeds}\n"
            f"Last seen: {last_seen}"
        )
        self.info_var.set(info)

        if p.thumbnail_path and Path(p.thumbnail_path).exists():
            try:
                img = Image.open(p.thumbnail_path)
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)
                self._photo_ref = photo
                self.thumb_label.configure(image=photo)
            except Exception:  # noqa: BLE001
                self.thumb_label.configure(image="")
        else:
            self.thumb_label.configure(image="")

    def _delete(self) -> None:
        p = self._selected_person()
        if p is None:
            return
        if not messagebox.askyesno("Confirm", f"Delete {p.name} and all their samples?", parent=self):
            return
        self.repo.delete_person(p.id)
        self._refresh()

    def _add_samples(self) -> None:
        p = self._selected_person()
        if p is None:
            return
        self.on_add_samples(p.id, p.name)
        self._refresh()

    def _sync_add_samples_state(self) -> None:
        if self._is_live_active():
            self._add_btn.state(["disabled"])
            self._live_hint.configure(text="Live recognition is using the camera. Stop it to add samples.")
        else:
            self._add_btn.state(["!disabled"])
            self._live_hint.configure(text="")

    def _tick_live_state(self) -> None:
        # Called while window is open. winfo_exists guards against races if the
        # window has been destroyed between schedules.
        if not self.winfo_exists():
            return
        self._sync_add_samples_state()
        self.after(500, self._tick_live_state)
