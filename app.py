import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import ttk

from utils.fusion import fuse_predictions
from utils.database import save_record

print("Starting App...")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Parkinson Detection System")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e1e2f")

        self.results = []
        self.stream = None

        main_frame = tk.Frame(root, bg="#1e1e2f")
        main_frame.pack(fill="both", expand=True)

        sidebar = tk.Frame(main_frame, bg="#2c2c3e", width=200)
        sidebar.pack(side="left", fill="y")

        content = tk.Frame(main_frame, bg="#1e1e2f")
        content.pack(side="right", fill="both", expand=True)

        title = tk.Label(
            content,
            text="Parkinson Detection Dashboard",
            font=("Arial", 18, "bold"),
            bg="#1e1e2f",
            fg="white"
        )
        title.pack(pady=10)

        text_frame = tk.Frame(content)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        self.text = tk.Text(
            text_frame,
            bg="#121212",
            fg="#00ffcc",
            font=("Consolas", 10),
            insertbackground="white",
            yscrollcommand=scrollbar.set
        )
        self.text.pack(fill="both", expand=True)

        scrollbar.config(command=self.text.yview)

        def add_btn(text, cmd):
            btn = tk.Button(
                sidebar,
                text=text,
                command=cmd,
                bg="#3a3a5a",
                fg="white",
                font=("Arial", 10, "bold"),
                relief="flat",
                padx=10,
                pady=8
            )
            btn.pack(pady=5, fill="x")

        add_btn("Upload Audio", self.audio_upload)
        add_btn("🎤 Start Recording", self.start_audio)
        add_btn("🛑 Stop Recording", self.stop_audio)

        add_btn("Handwriting", self.handwriting)
        add_btn("Tremor", self.tremor)
        add_btn("Face", self.face)
        add_btn("Gait", self.gait)

        add_btn("Final Prediction", self.final)

        add_btn("Load History", self.load_patient)
        add_btn("Progress Graph", self.show_graph)
        add_btn("Trend", self.show_trend)

    def log(self, msg):
        self.text.insert(tk.END, msg + "\n")

    # 🔥🔥 UPDATED SELECT PATIENT (FINAL VERSION)
    def select_patient(self):
        import json

        try:
            with open("patient_data.json", "r") as f:
                data = json.load(f)
            names = list(data.keys())
        except:
            names = []

        win = tk.Toplevel(self.root)
        win.title("Select or Add Patient")
        win.geometry("320x380")

        tk.Label(win, text="Patients", font=("Arial", 12, "bold")).pack(pady=5)

        listbox = tk.Listbox(win)
        listbox.pack(fill="both", expand=True, padx=10, pady=10)

        for name in names:
            listbox.insert(tk.END, name)

        selected_name = {"value": None}

        def select_existing():
            try:
                selected = listbox.get(listbox.curselection())
                selected_name["value"] = selected
                win.destroy()
            except:
                pass

        def add_new():
            new_name = simpledialog.askstring("New Patient", "Enter patient name:")
            if new_name:
                selected_name["value"] = new_name
                win.destroy()

        # 🔥 BUTTON AREA
        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="Select", command=select_existing).grid(row=0, column=0, padx=5)
        tk.Button(
            btn_frame,
            text="➕ Add New",
            command=add_new,
            bg="#4CAF50",
            fg="white"
        ).grid(row=0, column=1, padx=5)

        win.wait_window()

        return selected_name["value"]

    # 🔥 REST SAME (NO CHANGE BELOW)

    def audio_upload(self):
        from models.audio_model import predict_audio
        file = filedialog.askopenfilename()
        if not file:
            return
        p, explanation, counterfactual, confidence = predict_audio(file)
        self.results.append(p)
        self.log(f"Audio (File): {p:.3f}")
        self.log(f"Confidence: {confidence*100:.2f}%")
        self._show_audio_details(explanation, counterfactual)

    def start_audio(self):
        from models.audio_model import start_recording
        self.stream = start_recording()
        self.log("🎤 Recording started...")

    def stop_audio(self):
        from models.audio_model import stop_recording, noise_filter, show_waveform, play_audio, predict_audio
        if not self.stream:
            self.log("⚠️ Start recording first!")
            return
        file, audio = stop_recording(self.stream)
        self.stream = None
        self.log("🛑 Recording stopped")
        audio = noise_filter(audio)
        show_waveform(audio)
        play_audio(audio)
        p, explanation, counterfactual, confidence = predict_audio(file)
        self.results.append(p)
        self.log(f"Audio (Live): {p:.3f}")
        self.log(f"Confidence: {confidence*100:.2f}%")
        self._show_audio_details(explanation, counterfactual)

    def _show_audio_details(self, explanation, counterfactual):
        self.log("Top Features:")
        top = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for f, v in top:
            self.log(f"{f}: {v:.4f}")
        self.log("\nSuggestions to Improve:")
        for s in counterfactual:
            self.log(f"- {s}")

    def handwriting(self):
        from models.handwriting_model import predict_handwriting
        file = filedialog.askopenfilename()
        if not file:
            return
        p = predict_handwriting(file)
        self.results.append(p)
        self.log(f"Handwriting: {p:.3f}")

    def tremor(self):
        from models.tremor_model import predict_tremor
        p = predict_tremor()
        self.results.append(p)
        self.log(f"Tremor: {p:.3f}")

    def face(self):
        from models.face_model import predict_face
        p = predict_face()
        self.results.append(p)
        self.log(f"Face: {p*100:.2f}%")

    def gait(self):
        from models.gait_model import predict_gait
        p = predict_gait()
        self.results.append(p)
        self.log(f"Gait: {p:.3f}")

    def load_patient(self):
        import json
        name = self.select_patient()
        if not name:
            return
        try:
            with open("patient_data.json", "r") as f:
                data = json.load(f)
            if name in data:
                self.log(f"\nHistory for {name}:")
                for r in data[name]:
                    self.log(f"{r['date']} → {r['score']:.3f}")
        except:
            self.log("No database found")

    def show_graph(self):
        import json
        import matplotlib.pyplot as plt
        name = self.select_patient()
        if not name:
            return
        try:
            with open("patient_data.json", "r") as f:
                data = json.load(f)
            scores = [r["score"] for r in data[name]]
            dates = [r["date"] for r in data[name]]
            plt.figure()
            plt.plot(scores, marker='o')
            plt.title(f"{name} Progress")
            plt.xticks(range(len(dates)), dates, rotation=45)
            plt.grid()
            plt.tight_layout()
            plt.show()
        except:
            self.log("Error loading graph")

    def show_trend(self):
        import json
        name = self.select_patient()
        if not name:
            return
        try:
            with open("patient_data.json", "r") as f:
                data = json.load(f)
            scores = [r["score"] for r in data.get(name, [])]
            if len(scores) < 2:
                self.log("Not enough data")
                return
            if scores[-1] > scores[-2]:
                self.log("Trend: Worsening ⚠️")
            else:
                self.log("Trend: Improving ✅")
        except:
            self.log("Error loading trend")

    def final(self):
        if not self.results:
            self.log("No data!")
            return

        raw_score = float(fuse_predictions(self.results))
        name = self.select_patient()

        if not name:
            return

        from utils.database import get_patient_average
        history_avg = get_patient_average(name)

        final = 0.7 * raw_score + 0.3 * history_avg

        save_record(name, final)

        self.log(f"\nRaw Score: {raw_score:.3f}")
        self.log(f"History Avg: {history_avg:.3f}")
        self.log(f"FINAL (Personalized): {final:.3f}")

        if final > 0.75:
         self.log("🚨 CRITICAL ALERT!")
    
         import tkinter.messagebox as msg
         msg.showerror("CRITICAL ALERT", f"{name} is at HIGH RISK!")

        elif final > 0.4:
          self.log("⚠️ MODERATE")
        else:
           self.log("✅ LOW")

        self.results = []


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()