import customtkinter as ctk
import imaplib
import email
import time
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SpamScannerPro(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Email Spam Scanner Pro v7.0 - Academic Edition")
        self.geometry("1150x700") 
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.grid_columnconfigure((0, 1), weight=1) 
        self.grid_rowconfigure(3, weight=1) 

        # ==========================================
        # TOP SECTION: Controls & Progress
        # ==========================================
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, columnspan=2, pady=(15, 5), sticky="ew")
        self.header_frame.grid_columnconfigure(1, weight=1)

        self.title_label = ctk.CTkLabel(self.header_frame, text="Bulk Inbox AI Scanner", font=ctk.CTkFont(size=26, weight="bold"))
        self.title_label.grid(row=0, column=1, pady=5)

        # NEW FEATURE: Model Metrics Button
        self.metrics_btn = ctk.CTkButton(self.header_frame, text="📊 View Model Metrics", fg_color="#8E44AD", hover_color="#732D91", command=self.show_metrics, state="disabled")
        self.metrics_btn.grid(row=0, column=2, padx=20, sticky="e")

        self.sync_frame = ctk.CTkFrame(self)
        self.sync_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        self.sync_frame.grid_columnconfigure((0, 1), weight=2) 
        self.sync_frame.grid_columnconfigure(2, weight=1)    

        self.email_entry = ctk.CTkEntry(self.sync_frame, placeholder_text="Gmail Address", height=35)
        self.email_entry.grid(row=0, column=0, padx=5, pady=15, sticky="ew")

        self.pass_entry = ctk.CTkEntry(self.sync_frame, placeholder_text="16-Letter App Password", show="*", height=35)
        self.pass_entry.grid(row=0, column=1, padx=5, pady=15, sticky="ew")

        self.count_entry = ctk.CTkEntry(self.sync_frame, placeholder_text="How many emails?", height=35)
        self.count_entry.grid(row=0, column=2, padx=5, pady=15, sticky="ew")

        self.scan_btn = ctk.CTkButton(self.sync_frame, text="Start Scan", font=ctk.CTkFont(weight="bold"), height=35, command=self.start_batch_scan, state="disabled")
        self.scan_btn.grid(row=0, column=3, padx=10, pady=15)

        self.progress_bar = ctk.CTkProgressBar(self, width=600, height=15)
        self.progress_bar.grid(row=2, column=0, columnspan=2, padx=20, pady=(10, 0))
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(self, text="⚙️ Initializing ML Pipeline... Please wait.", font=ctk.CTkFont(size=16, weight="bold"), text_color="#F39C12")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(5, 10), sticky="n")

        # ==========================================
        # BOTTOM SECTION: Categorized Lists
        # ==========================================
        self.safe_frame = ctk.CTkFrame(self)
        self.safe_frame.grid(row=4, column=0, padx=(20, 10), pady=(0, 20), sticky="nsew")
        self.safe_frame.grid_rowconfigure(1, weight=1)
        self.safe_frame.grid_columnconfigure(0, weight=1)

        self.safe_title = ctk.CTkLabel(self.safe_frame, text="✅ Safe Inbox (Ham)", font=ctk.CTkFont(size=18, weight="bold"), text_color="#2ECC71")
        self.safe_title.grid(row=0, column=0, pady=10)

        self.safe_log = ctk.CTkTextbox(self.safe_frame, font=ctk.CTkFont(size=13))
        self.safe_log.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.safe_log.insert("0.0", "Awaiting scan...\n")
        self.safe_log.configure(state="disabled")

        self.spam_frame = ctk.CTkFrame(self)
        self.spam_frame.grid(row=4, column=1, padx=(10, 20), pady=(0, 20), sticky="nsew")
        self.spam_frame.grid_rowconfigure(1, weight=1)
        self.spam_frame.grid_columnconfigure(0, weight=1)

        self.spam_title = ctk.CTkLabel(self.spam_frame, text="⚠️ Spam Quarantine", font=ctk.CTkFont(size=18, weight="bold"), text_color="#E74C3C")
        self.spam_title.grid(row=0, column=0, pady=10)

        self.spam_log = ctk.CTkTextbox(self.spam_frame, font=ctk.CTkFont(size=13))
        self.spam_log.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.spam_log.insert("0.0", "Awaiting scan...\n")
        self.spam_log.configure(state="disabled")

        self.grid_rowconfigure(4, weight=1) 

        # Trigger background loading
        self.after(100, self.initialize_ml_backend)


    # --- Application Logic ---

    def initialize_ml_backend(self):
        """Handles Model Persistence (MLOps) by loading saved models, or training new ones if none exist."""
        model_file = 'spam_model.pkl'
        vectorizer_file = 'vectorizer.pkl'
        metrics_file = 'model_metrics.pkl'

        if os.path.exists(model_file) and os.path.exists(vectorizer_file):
            # MLOps: Load pre-trained model directly from disk
            self.model = joblib.load(model_file)
            self.vectorizer = joblib.load(vectorizer_file)
            self.metrics = joblib.load(metrics_file)
            self.status_label.configure(text="✅ Pre-trained AI Loaded from Disk. Ready to scan.", text_color="#2ECC71")
        else:
            # Training Pipeline
            self.status_label.configure(text="⚙️ First Run: Downloading 5,500+ emails and training AI...", text_color="#F39C12")
            self.update()
            
            try:
                url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
                df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
            except:
                self.status_label.configure(text="❌ Failed to download dataset. Check internet connection.", text_color="#E74C3C")
                return

            # Train/Test Split for rigorous evaluation
            X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
            
            self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=2)
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            self.model = MultinomialNB(alpha=0.1)
            self.model.fit(X_train_vec, y_train)

            # Evaluate Model
            predictions = self.model.predict(X_test_vec)
            conf_matrix = confusion_matrix(y_test, predictions, labels=['ham', 'spam'])
            class_report = classification_report(y_test, predictions)

            self.metrics = {
                'conf_matrix': conf_matrix,
                'class_report': class_report
            }

            # Save to disk
            joblib.dump(self.model, model_file)
            joblib.dump(self.vectorizer, vectorizer_file)
            joblib.dump(self.metrics, metrics_file)

            self.status_label.configure(text=f"✅ AI Trained & Saved (Accuracy computed). Ready to scan.", text_color="#2ECC71")

        self.scan_btn.configure(state="normal")
        self.metrics_btn.configure(state="normal")


    def show_metrics(self):
        """Displays the Confusion Matrix and Classification Report in a professional pop-up window."""
        metrics_window = ctk.CTkToplevel(self)
        metrics_window.title("Model Evaluation Metrics")
        metrics_window.geometry("800x550")
        metrics_window.focus()

        title_lbl = ctk.CTkLabel(metrics_window, text="Algorithm Evaluation Report", font=ctk.CTkFont(size=20, weight="bold"))
        title_lbl.pack(pady=10)

        # Matplotlib Figure for Confusion Matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(self.metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham (Safe)', 'Spam'], yticklabels=['Ham (Safe)', 'Spam'], ax=ax)
        ax.set_ylabel('Actual Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix (Test Set)')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=metrics_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side="left", fill="both", expand=True, padx=20, pady=20)

        # Classification Report Text
        report_frame = ctk.CTkFrame(metrics_window)
        report_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        report_title = ctk.CTkLabel(report_frame, text="Classification Report", font=ctk.CTkFont(weight="bold"))
        report_title.pack(pady=10)

        report_textbox = ctk.CTkTextbox(report_frame, font=ctk.CTkFont(family="Courier", size=14))
        report_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        report_textbox.insert("0.0", self.metrics['class_report'])
        report_textbox.configure(state="disabled")


    def start_batch_scan(self):
        username = self.email_entry.get().strip()
        app_password = self.pass_entry.get().strip().replace(" ", "") 
        count_input = self.count_entry.get().strip()

        if not username or not app_password:
            self.status_label.configure(text="❌ Error: Please enter your Gmail and App Password.", text_color="#E74C3C")
            return

        if not count_input.isdigit() or int(count_input) <= 0:
            self.status_label.configure(text="❌ Error: Please enter a valid number of emails to scan.", text_color="#E74C3C")
            return
            
        requested_amount = int(count_input)

        self.scan_btn.configure(state="disabled")
        self.progress_bar.set(0)
        
        self.safe_log.configure(state="normal")
        self.spam_log.configure(state="normal")
        self.safe_log.delete("0.0", "end")
        self.spam_log.delete("0.0", "end")
        self.safe_log.configure(state="disabled")
        self.spam_log.configure(state="disabled")

        self.status_label.configure(text="🔄 Connecting to IMAP Server...", text_color="#3498DB")
        self.update() 

        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(username, app_password)
            mail.select("inbox")

            status, messages = mail.search(None, "ALL")
            all_ids = messages[0].split()
            total_available = len(all_ids)
            
            if total_available == 0:
                self.status_label.configure(text="Inbox is completely empty.", text_color="#ffffff")
                self.scan_btn.configure(state="normal")
                return
                
            if requested_amount > total_available:
                self.status_label.configure(
                    text=f"❌ Error: You only have {total_available} emails in your inbox. Please choose {total_available} or less.", 
                    text_color="#E74C3C"
                )
                self.scan_btn.configure(state="normal")
                return
                
            target_ids = all_ids[-requested_amount:]
            safe_count = 0
            spam_count = 0
            
            # Extract feature names once for Explainable AI (XAI)
            feature_names = np.array(self.vectorizer.get_feature_names_out())

            for index, email_id in enumerate(target_ids):
                self.status_label.configure(text=f"🔍 Scanning Email {index + 1} of {requested_amount}...", text_color="#F39C12")
                self.update() 
                
                res, msg_data = mail.fetch(email_id, "(RFC822)")
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)
                
                subject = msg["Subject"] if msg["Subject"] else "(No Subject)"
                    
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            break
                else:
                    body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')

                text_to_analyze = subject + " " + body
                
                if text_to_analyze.strip():
                    new_email_vec = self.vectorizer.transform([text_to_analyze])
                    prediction = self.model.predict(new_email_vec)[0]
                else:
                    prediction = "ham"

                # Base log entry
                log_entry = f"Subject: {subject}\nSnippet: {body[:50].replace(chr(10), ' ')}...\n"
                
                # --- EXPLAINABLE AI (XAI) LOGIC ---
                if prediction == 'spam':
                    spam_count += 1
                    
                    # Find the top 3 highest weighted words that triggered the spam classification
                    tfidf_sorting = np.argsort(new_email_vec.toarray()).flatten()[::-1]
                    top_n_words = feature_names[tfidf_sorting][:3]
                    triggers = ", ".join(top_n_words)
                    
                    log_entry += f"🚩 AI Triggers: [{triggers}]\n" + ("-"*40) + "\n\n"
                    
                    self.spam_log.configure(state="normal")
                    self.spam_log.insert("end", log_entry)
                    self.spam_log.configure(state="disabled")
                else:
                    safe_count += 1
                    log_entry += ("-"*40) + "\n\n"
                    
                    self.safe_log.configure(state="normal")
                    self.safe_log.insert("end", log_entry)
                    self.safe_log.configure(state="disabled")

                progress = (index + 1) / requested_amount
                self.progress_bar.set(progress)
                time.sleep(0.4) 
                self.update() 

            self.status_label.configure(text=f"✅ Scan Complete! Found {safe_count} Safe and {spam_count} Spam.", text_color="#2ECC71")
            self.scan_btn.configure(state="normal")

        except Exception as e:
            self.status_label.configure(text=f"❌ Error during scan: {str(e)}", text_color="#E74C3C")
            self.scan_btn.configure(state="normal")

if __name__ == "__main__":
    app = SpamScannerPro()
    app.mainloop()