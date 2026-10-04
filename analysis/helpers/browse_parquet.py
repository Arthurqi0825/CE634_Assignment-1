"""Print Parquet schemas, then browse selected columns and row numbers in a window.

Examples (run from the project root):
    .venv/bin/python -m analysis.helpers.browse_parquet
    .venv/bin/python -m analysis.helpers.browse_parquet path/to/file.parquet
    .venv/bin/python -m analysis.helpers.browse_parquet --no-gui
"""

from __future__ import annotations

import argparse
import tkinter as tk
from bisect import bisect_right
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from tkinter import filedialog, messagebox, ttk

import pyarrow.parquet as pq

from analysis.common import PROJECT_ROOT, RAW_DIR

ROOT = PROJECT_ROOT
DEFAULT_DIR = RAW_DIR
PAGE_SIZES = (50, 100, 250, 500)


def open_parquet(path: Path) -> pq.ParquetFile:
    if not path.is_file() or path.suffix.lower() != ".parquet":
        raise ValueError(f"Not a readable .parquet file: {path}")
    return pq.ParquetFile(path)


def print_schema(path: Path, parquet: pq.ParquetFile) -> None:
    schema = parquet.schema_arrow
    print(f"\n{path}")
    print(f"Rows: {parquet.metadata.num_rows:,} | Columns: {len(schema)}")
    print(" No.  Key                              Type")
    print("----  -------------------------------  ------------------------------")
    for number, field in enumerate(schema, 1):
        print(f"{number:>4}  {field.name:<31}  {field.type}")
    print(flush=True)


def row_group_starts(parquet: pq.ParquetFile) -> list[int]:
    starts = [0]
    for index in range(parquet.num_row_groups):
        starts.append(starts[-1] + parquet.metadata.row_group(index).num_rows)
    return starts


def read_page(path: Path, columns: list[str], start: int, size: int) -> list[tuple]:
    """Read only requested columns; convert only visible rows to Python objects."""
    parquet = open_parquet(path)
    starts = row_group_starts(parquet)
    stop = min(start + size, starts[-1])
    rows: list[tuple] = []
    position = start
    while position < stop:
        group = bisect_right(starts, position) - 1
        local_start = position - starts[group]
        local_stop = min(stop, starts[group + 1]) - starts[group]
        batch_start = 0
        for batch in parquet.iter_batches(
            row_groups=[group], columns=columns, batch_size=65536
        ):
            batch_stop = batch_start + batch.num_rows
            if batch_stop > local_start and batch_start < local_stop:
                sliced = batch.slice(
                    max(local_start - batch_start, 0),
                    min(local_stop, batch_stop) - max(local_start, batch_start),
                )
                values = [column.to_pylist() for column in sliced.columns]
                rows.extend(zip(*values))
            batch_start = batch_stop
            if batch_start >= local_stop:
                break
        position = starts[group] + local_stop
    return rows


def display_value(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class Browser:
    def __init__(self, root: tk.Tk, files: list[Path]) -> None:
        self.root = root
        self.files = files
        self.path = files[0]
        self.parquet = open_parquet(self.path)
        self.start = 0
        self.columns: list[str] = []
        self.visible_rows: list[tuple] = []
        self.busy = False
        self.results: Queue[tuple[str, object]] = Queue()

        root.title("Parquet 行浏览器")
        root.geometry("1250x760")
        root.minsize(760, 480)

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="文件：").pack(side="left")
        self.file_var = tk.StringVar(value=str(self.path))
        self.file_box = ttk.Combobox(
            top,
            textvariable=self.file_var,
            values=[str(p) for p in files],
            state="readonly",
        )
        self.file_box.pack(side="left", fill="x", expand=True)
        self.file_box.bind("<<ComboboxSelected>>", self.change_file)
        self.browse_button = ttk.Button(top, text="打开其他文件…", command=self.browse_file)
        self.browse_button.pack(side="left", padx=(8, 0))

        body = ttk.PanedWindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8)
        side = ttk.Frame(body, width=250)
        body.add(side, weight=0)
        table_area = ttk.Frame(body)
        body.add(table_area, weight=1)

        self.info_var = tk.StringVar()
        ttk.Label(side, textvariable=self.info_var, wraplength=235).pack(
            fill="x", pady=(0, 8)
        )
        ttk.Label(side, text="列名（Ctrl/Shift 可多选）：").pack(anchor="w")
        list_frame = ttk.Frame(side)
        list_frame.pack(fill="both", expand=True)
        self.column_list = tk.Listbox(
            list_frame, selectmode="extended", exportselection=False
        )
        self.column_list.pack(side="left", fill="both", expand=True)
        list_scroll = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.column_list.yview
        )
        list_scroll.pack(side="right", fill="y")
        self.column_list.configure(yscrollcommand=list_scroll.set)
        select_buttons = ttk.Frame(side)
        select_buttons.pack(fill="x", pady=5)
        ttk.Button(select_buttons, text="全选", command=self.select_all).pack(side="left")
        ttk.Button(select_buttons, text="清空", command=self.select_none).pack(
            side="left", padx=5
        )
        self.apply_button = ttk.Button(side, text="显示选中列", command=self.apply_columns)
        self.apply_button.pack(fill="x")

        controls = ttk.Frame(table_area)
        controls.pack(fill="x", pady=(0, 6))
        ttk.Label(controls, text="起始行（从 1 开始）：").pack(side="left")
        self.row_var = tk.StringVar(value="1")
        self.row_entry = ttk.Entry(controls, textvariable=self.row_var, width=13)
        self.row_entry.pack(side="left")
        self.row_entry.bind("<Return>", self.go_to_row)
        self.go_button = ttk.Button(controls, text="跳转", command=self.go_to_row)
        self.go_button.pack(side="left", padx=4)
        ttk.Label(controls, text="每页：").pack(side="left", padx=(12, 0))
        self.size_var = tk.StringVar(value="100")
        self.size_box = ttk.Combobox(
            controls,
            textvariable=self.size_var,
            values=PAGE_SIZES,
            width=5,
            state="readonly",
        )
        self.size_box.pack(side="left")
        self.size_box.bind("<<ComboboxSelected>>", self.reload_page)
        self.prev_button = ttk.Button(controls, text="上一页", command=self.previous_page)
        self.prev_button.pack(side="left", padx=(12, 4))
        self.next_button = ttk.Button(controls, text="下一页", command=self.next_page)
        self.next_button.pack(side="left")

        grid = ttk.Frame(table_area)
        grid.pack(fill="both", expand=True)
        self.table = ttk.Treeview(grid, show="headings")
        self.table.grid(row=0, column=0, sticky="nsew")
        self.table.bind("<Double-1>", self.show_row_details)
        yscroll = ttk.Scrollbar(grid, orient="vertical", command=self.table.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(grid, orient="horizontal", command=self.table.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.table.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        grid.rowconfigure(0, weight=1)
        grid.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar()
        ttk.Label(root, textvariable=self.status_var, padding=(8, 5)).pack(fill="x")
        self.setup_file()
        root.after(100, self.poll_results)

    @property
    def total(self) -> int:
        return self.parquet.metadata.num_rows

    @property
    def page_size(self) -> int:
        return int(self.size_var.get())

    def setup_file(self) -> None:
        self.info_var.set(
            f"{self.path.name}\n{self.total:,} 行 · {len(self.parquet.schema_arrow)} 列"
        )
        self.column_list.delete(0, "end")
        for field in self.parquet.schema_arrow:
            self.column_list.insert("end", f"{field.name}  ({field.type})")
        self.select_all()
        self.start = 0
        if self.total == 0:
            self.table.delete(*self.table.get_children())
            self.visible_rows = []
            self.status_var.set("此文件没有数据行。")
            self.prev_button.configure(state="disabled")
            self.next_button.configure(state="disabled")
            return
        self.load_page()

    def select_all(self) -> None:
        self.column_list.select_set(0, "end")

    def select_none(self) -> None:
        self.column_list.selection_clear(0, "end")

    def change_file(self, _event=None) -> None:
        self.open_file(Path(self.file_var.get()))

    def browse_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="选择 Parquet 文件",
            initialdir=self.path.parent,
            filetypes=[("Parquet files", "*.parquet"), ("All files", "*")],
        )
        if chosen:
            self.open_file(Path(chosen))

    def open_file(self, path: Path) -> None:
        try:
            parquet = open_parquet(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法打开文件", str(exc))
            self.file_var.set(str(self.path))
            return
        self.path, self.parquet = path, parquet
        if path not in self.files:
            self.files.append(path)
            self.file_box.configure(values=[str(p) for p in self.files])
        self.file_var.set(str(path))
        print_schema(path, parquet)
        self.setup_file()

    def apply_columns(self) -> None:
        if not self.column_list.curselection():
            messagebox.showinfo("请选择列", "至少选择一列后再显示。")
            return
        self.load_page()

    def go_to_row(self, _event=None) -> None:
        try:
            number = int(self.row_var.get().strip().replace(",", ""))
        except ValueError:
            messagebox.showerror("行号无效", "请输入整数行号。")
            return
        if not 1 <= number <= self.total:
            messagebox.showerror("行号超出范围", f"请输入 1 到 {self.total:,} 之间的行号。")
            return
        self.start = number - 1
        self.load_page()

    def previous_page(self) -> None:
        self.start = max(0, self.start - self.page_size)
        self.load_page()

    def next_page(self) -> None:
        if self.start + self.page_size < self.total:
            self.start += self.page_size
            self.load_page()

    def reload_page(self, _event=None) -> None:
        self.load_page()

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for widget in (
            self.browse_button,
            self.apply_button,
            self.go_button,
            self.prev_button,
            self.next_button,
            self.row_entry,
        ):
            widget.configure(state=state)
        self.file_box.configure(state="disabled" if busy else "readonly")
        self.size_box.configure(state="disabled" if busy else "readonly")

    def show_row_details(self, event) -> None:
        item = self.table.identify_row(event.y)
        if not item:
            return
        offset = self.table.index(item)
        if offset >= len(self.visible_rows):
            return
        dialog = tk.Toplevel(self.root)
        dialog.title(f"第 {self.start + offset + 1:,} 行")
        dialog.geometry("680x520")
        frame = ttk.Frame(dialog, padding=8)
        frame.pack(fill="both", expand=True)
        details = tk.Text(frame, wrap="none")
        details.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=details.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=details.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        details.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        details.insert("end", f"行号：{self.start + offset + 1:,}\n\n")
        for name, value in zip(self.columns, self.visible_rows[offset]):
            field_type = self.parquet.schema_arrow.field(name).type
            details.insert(
                "end", f"{name} ({field_type})\n  {display_value(value)}\n\n"
            )
        details.configure(state="disabled")

    def load_page(self) -> None:
        if self.busy or self.total == 0:
            return
        indices = self.column_list.curselection()
        self.columns = [self.parquet.schema_arrow[index].name for index in indices]
        if not self.columns:
            return
        self.row_var.set(str(self.start + 1))
        self.status_var.set(f"正在读取第 {self.start + 1:,} 行起的数据…")
        self.set_busy(True)
        path, columns, start, size = (
            self.path,
            self.columns[:],
            self.start,
            self.page_size,
        )

        def worker() -> None:
            try:
                self.results.put(("ok", read_page(path, columns, start, size)))
            except Exception as exc:
                self.results.put(("error", exc))

        Thread(target=worker, daemon=True).start()

    def poll_results(self) -> None:
        try:
            kind, payload = self.results.get_nowait()
        except Empty:
            pass
        else:
            self.set_busy(False)
            if kind == "error":
                self.status_var.set("读取失败")
                messagebox.showerror("读取 Parquet 失败", str(payload))
            else:
                rows = payload
                self.visible_rows = rows
                names = ["行号", *self.columns]
                ids = [f"c{i}" for i in range(len(names))]
                self.table.configure(columns=ids)
                for ident, name in zip(ids, names):
                    self.table.heading(ident, text=name)
                    self.table.column(
                        ident,
                        width=95 if name == "行号" else 160,
                        minwidth=75,
                        stretch=False,
                        anchor="w",
                    )
                self.table.delete(*self.table.get_children())
                for offset, values in enumerate(rows):
                    self.table.insert(
                        "",
                        "end",
                        values=(
                            f"{self.start + offset + 1:,}",
                            *(display_value(value) for value in values),
                        ),
                    )
                end = self.start + len(rows)
                self.status_var.set(
                    f"显示第 {self.start + 1:,}–{end:,} 行，共 {self.total:,} 行；"
                    "可滚动、翻页或跳转；双击行查看完整字段。"
                )
                self.prev_button.configure(state="normal" if self.start else "disabled")
                self.next_button.configure(
                    state="normal" if end < self.total else "disabled"
                )
        self.root.after(100, self.poll_results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="*", type=Path, help="要浏览的 Parquet 文件；默认项目中的四个原始文件"
    )
    parser.add_argument("--no-gui", action="store_true", help="仅打印列名和类型")
    args = parser.parse_args()
    files = args.files or sorted(DEFAULT_DIR.glob("*.parquet"))
    if not files:
        parser.error(f"没有找到 Parquet 文件：{DEFAULT_DIR}")
    for path in files:
        try:
            print_schema(path, open_parquet(path))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if args.no_gui:
        return
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(1, f"无法打开图形窗口：{exc}\n请在桌面环境运行，或使用 --no-gui。\n")
    Browser(root, files)
    root.mainloop()


if __name__ == "__main__":
    main()
