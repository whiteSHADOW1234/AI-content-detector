import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# 定義可選的模型和其標籤
MODEL_OPTIONS = {
    "GPT-2 Small (117M)": "gpt2",
    "GPT-2 Medium (345M)": "gpt2-medium",
    "GPT-2 Large (774M)": "gpt2-large"
}

# --- 新增：定義顏色梯度 ---
# 這些顏色標籤必須在 ScrolledText widget 中配置
LOSS_TAGS = {
    "low_loss": '#e6ffe6',   # 非常容易預測 (偏AI)
    "med_low_loss": '#ffffcc', # 容易預測
    "medium_loss": '#ffebcc', # 中等難度
    "high_loss": '#ffcccc',   # 較難預測
    "very_high_loss": '#ff8080' # 非常難預測 (偏人)
}

class PerplexityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPT-2 文章困惑度分析 (含損失熱力圖)")
        self.root.geometry("850x750") 

        self.model = None
        self.tokenizer = None
        self.current_model_name = MODEL_OPTIONS["GPT-2 Small (117M)"] 

        self._create_widgets()
        self._load_model_async()

    def _clear_input(self):
        """清空輸入文本框的內容"""
        self.input_text.delete("1.0", tk.END)
        self.input_text.tag_remove(tk.ALL, "1.0", tk.END) # 清除所有高亮

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="15 15 15 15")
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)

        # 1. 模型選擇和狀態區域
        config_frame = ttk.Frame(main_frame)
        config_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)
        
        ttk.Label(config_frame, text="選擇模型大小:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        self.model_var = tk.StringVar(value="GPT-2 Small (117M)")
        self.model_combobox = ttk.Combobox(
            config_frame, 
            textvariable=self.model_var, 
            values=list(MODEL_OPTIONS.keys()),
            state="readonly",
            width=25
        )
        self.model_combobox.grid(row=0, column=1, sticky="w", padx=(0, 10))
        self.model_combobox.bind("<<ComboboxSelected>>", self._on_model_select_change)

        self.status_var = tk.StringVar()
        self.status_var.set("正在載入模型，請稍候...")
        self.status_label = ttk.Label(config_frame, textvariable=self.status_var, font=('Arial', 10, 'italic'), foreground='blue')
        self.status_label.grid(row=0, column=2, sticky="e")
        config_frame.columnconfigure(2, weight=1) 

        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate', length=200)
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(0, 10))


        # 2. 輸入文本區域控制
        input_controls_frame = ttk.Frame(main_frame)
        input_controls_frame.grid(row=2, column=0, sticky="ew")
        input_controls_frame.columnconfigure(0, weight=1) 
        
        ttk.Label(input_controls_frame, text="請輸入您的英文文章：", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        clear_button = ttk.Button(input_controls_frame, text="清空文本", command=self._clear_input)
        clear_button.grid(row=0, column=1, sticky="e", padx=(10, 0))

        self.input_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=80, height=15, font=('Arial', 10))
        self.input_text.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        
        # --- 設定 ScrolledText 的 Tag 樣式 (關鍵步驟) ---
        for tag_name, color in LOSS_TAGS.items():
            self.input_text.tag_config(tag_name, background=color)

        self.input_text.bind("<Control-a>", self.select_all_text)
        self.input_text.bind("<Command-a>", self.select_all_text) 

        # 3. 計算按鈕
        self.calculate_button = ttk.Button(main_frame, text="計算並高亮顯示困惑度", command=self._start_calculation, state=tk.DISABLED)
        self.calculate_button.grid(row=4, column=0, sticky="ew", pady=(0, 20))

        # 4. 結果顯示區域
        ttk.Label(main_frame, text="=== 分析結果 ===", font=('Arial', 12, 'bold')).grid(row=5, column=0, sticky="w", pady=(10, 5))

        result_frame = ttk.Frame(main_frame)
        result_frame.grid(row=6, column=0, sticky="ew", padx=10)
        result_frame.columnconfigure(0, weight=1) 

        self.token_count_var = tk.StringVar(value="")
        ttk.Label(result_frame, text="分析的 Token 總數：", font=('Arial', 11)).grid(row=0, column=0, sticky="w")
        ttk.Label(result_frame, textvariable=self.token_count_var, font=('Arial', 11, 'bold'), foreground='darkcyan').grid(row=0, column=1, sticky="e")
        
        self.ppl_var = tk.StringVar(value="")
        ttk.Label(result_frame, text="整體平均困惑度（PPL）：", font=('Arial', 11)).grid(row=1, column=0, sticky="w")
        ttk.Label(result_frame, textvariable=self.ppl_var, font=('Arial', 11, 'bold'), foreground='darkgreen').grid(row=1, column=1, sticky="e")

        self.var_loss_var = tk.StringVar(value="")
        ttk.Label(result_frame, text="Token 損失變異量：", font=('Arial', 11)).grid(row=2, column=0, sticky="w")
        ttk.Label(result_frame, textvariable=self.var_loss_var, font=('Arial', 11, 'bold'), foreground='darkgreen').grid(row=2, column=1, sticky="e")

        ttk.Separator(main_frame, orient='horizontal').grid(row=7, column=0, sticky="ew", pady=(10, 10))

        # 判斷結果
        self.prediction_var = tk.StringVar(value="")
        ttk.Label(main_frame, text="判斷結果：", font=('Arial', 12)).grid(row=8, column=0, sticky="w", padx=(10, 0))
        self.prediction_label = ttk.Label(main_frame, textvariable=self.prediction_var, font=('Arial', 14, 'bold'))
        self.prediction_label.grid(row=8, column=0, sticky="e", padx=(10, 10))

        # --- 5. 新增高亮圖例 (Legend) ---
        ttk.Label(main_frame, text="Token 預測難度熱力圖：", font=('Arial', 10, 'bold')).grid(row=9, column=0, sticky="w", pady=(10, 5))
        
        legend_frame = ttk.Frame(main_frame)
        legend_frame.grid(row=10, column=0, sticky="ew", pady=(0, 5))
        
        legend_text = [
            ("易預測 (AI 傾向):", "low_loss"),
            ("中等難度:", "medium_loss"),
            ("難預測 (人類傾向):", "very_high_loss")
        ]
        
        col_idx = 0
        for text, tag in legend_text:
            l = ttk.Label(legend_frame, text=f"■ {text}", background=LOSS_TAGS[tag], borderwidth=1, relief="solid", padding=(5, 2))
            l.grid(row=0, column=col_idx, padx=5, sticky="w")
            col_idx += 1
            legend_frame.columnconfigure(col_idx - 1, weight=1)

        # 底部說明
        ttk.Label(main_frame, text="提示：文本中的顏色標記顯示了模型對每個詞語預測的難度（損失值）。", font=('Arial', 9, 'italic')).grid(row=11, column=0, sticky="w", pady=(10, 0))
        ttk.Label(main_frame, text="注意：這些判斷閾值是經驗性的，可能需要根據實際應用調整。", font=('Arial', 9, 'italic')).grid(row=12, column=0, sticky="w", pady=(0, 5))

        main_frame.rowconfigure(3, weight=1) 

    def select_all_text(self, event=None):
        self.input_text.tag_add("sel", "1.0", "end-1c")
        return "break" 

    def _on_model_select_change(self, event):
        selected_key = self.model_var.get()
        new_model_name = MODEL_OPTIONS.get(selected_key)
        
        if new_model_name != self.current_model_name:
            self.current_model_name = new_model_name
            self._load_model_async()

    def _load_model_async(self):
        # ... (模型加載邏輯不變) ...
        self.calculate_button.config(state=tk.DISABLED)
        self.status_var.set(f"正在載入 {self.current_model_name} 模型... 請稍候...")
        self.progress_bar.start(10) 
        self.root.update_idletasks() 

        def load_task():
            try:
                model = GPT2LMHeadModel.from_pretrained(self.current_model_name)
                tokenizer = GPT2TokenizerFast.from_pretrained(self.current_model_name)
                model.eval()
                
                self.model = model
                self.tokenizer = tokenizer
                
                self.root.after(0, self._on_model_loaded, True) 
            except Exception as e:
                self.root.after(0, self._on_model_loaded, False, str(e))

        threading.Thread(target=load_task).start()

    def _on_model_loaded(self, success, error_message=None):
        self.progress_bar.stop()
        if success:
            self.status_var.set(f"模型 {self.current_model_name} 載入完成。")
            self.calculate_button.config(state=tk.NORMAL)
        else:
            self.status_var.set(f"模型載入失敗: {error_message}")
            messagebox.showerror("錯誤", f"無法載入模型：{error_message}\n請檢查網絡連接或模型名稱。")
            self.calculate_button.config(state=tk.DISABLED)

    def _start_calculation(self):
        text = self.input_text.get("1.0", "end-1c").strip() 

        if not text:
            messagebox.showwarning("輸入錯誤", "請在文本框中輸入文章後再計算。")
            return

        # 計算前清除所有舊的高亮標籤
        self.input_text.tag_remove(tk.ALL, "1.0", tk.END)

        self.calculate_button.config(state=tk.DISABLED)
        self.status_var.set("正在計算中...請稍候...")
        self.ppl_var.set("")
        self.var_loss_var.set("")
        self.token_count_var.set("") 
        self.prediction_var.set("")
        self.progress_bar.start(50) 
        self.root.update_idletasks() 

        def calculation_task():
            try:
                avg_ppl, var_token_losses, token_count, prediction_text, offsets, token_losses_np = self._calculate_perplexity(text)
                
                # 將計算結果和高亮所需數據傳遞給主執行緒
                self.root.after(0, self._on_calculation_complete, avg_ppl, var_token_losses, token_count, prediction_text, offsets, token_losses_np)
            except Exception as e:
                self.root.after(0, self._on_calculation_error, str(e))

        threading.Thread(target=calculation_task).start()

    def _on_calculation_complete(self, avg_ppl, var_token_losses, token_count, prediction_text, offsets, token_losses_np):
        """計算完成後更新 GUI 和應用高亮"""
        self.progress_bar.stop()
        
        self.token_count_var.set(f"{token_count}")
        self.ppl_var.set(f"{avg_ppl:.2f}" if avg_ppl != float('inf') else "N/A (文本過短)")
        self.var_loss_var.set(f"{var_token_losses:.2f}" if var_token_losses != float('inf') else "N/A (文本過短)")
        self.prediction_var.set(prediction_text)
        self.status_var.set("計算完成，請查看高亮結果。")
        self.calculate_button.config(state=tk.NORMAL)

        # 應用高亮 (深度學習可視化部分)
        if token_losses_np is not None:
            self._highlight_losses(offsets, token_losses_np)

        # 根據預測結果調整預測文字的顏色
        if "極高可能是AI生成內容" in prediction_text:
            self.prediction_label.config(foreground='red')
        # ... (其餘顏色邏輯不變) ...
        elif "可能是AI生成" in prediction_text:
            self.prediction_label.config(foreground='orange')
        elif "較可能是人類撰寫" in prediction_text:
            self.prediction_label.config(foreground='darkblue')
        elif "極高可能是人類撰寫" in prediction_text:
            self.prediction_label.config(foreground='green')
        else:
            self.prediction_label.config(foreground='purple') 


    def _highlight_losses(self, offsets, token_losses):
        """
        將 Token 損失值映射到顏色標籤，並應用於文本框。
        這直接反映了模型對每個詞彙的預測信心。
        """
        if not offsets or not token_losses.size:
            return

        # 設置損失值的閾值（這些是經驗性的，可以根據模型輸出分佈調整）
        # 損失值通常在 0 到 10 之間，但極端值可能更高。
        q1 = np.percentile(token_losses, 20)
        q2 = np.percentile(token_losses, 50)
        q3 = np.percentile(token_losses, 80)
        q4 = np.percentile(token_losses, 95)
        
        # 迭代從第二個 Token 開始（第一個 Token 沒有前文，無法計算損失）
        # token_losses 的長度比 offsets 少 1
        for i in range(len(token_losses)):
            loss = token_losses[i]
            
            # offsets 是針對整個輸入序列的，我們需要取從第二個 Token 開始的 offset
            # 因為損失是對應於 "shift_labels" (即 token[1:])
            start_char, end_char = offsets[i + 1] 

            # 確保偏移量是有效的（非空白符）
            if start_char == end_char:
                continue
            
            # 將字元索引轉換為 tkinter 的文本索引 (1.0, 1.5, etc.)
            start_index = f"1.0 + {start_char}c"
            end_index = f"1.0 + {end_char}c"

            # 映射損失到顏色標籤
            if loss <= q1:
                tag = "low_loss" # 綠色：極容易預測 (AI傾向)
            elif loss <= q2:
                tag = "med_low_loss"
            elif loss <= q3:
                tag = "medium_loss"
            elif loss <= q4:
                tag = "high_loss"
            else:
                tag = "very_high_loss" # 紅色：極難預測 (人類傾向)

            self.input_text.tag_add(tag, start_index, end_index)


    def _calculate_perplexity(self, text):
        """
        核心困惑度計算邏輯，現在額外返回 `offsets` 和 `token_losses_np`。
        """
        # 使用 return_offsets_mapping=True 來獲取每個 Token 對應的原始文本字元範圍
        inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        input_ids = inputs["input_ids"]
        
        # offsets mapping: [batch_size, seq_len, 2]
        offsets = inputs["offset_mapping"][0].tolist() 

        token_count = input_ids.shape[1] 
        avg_ppl = float('inf')
        var_token_losses = float('inf')
        prediction_text = "無法判斷 (文本過短或錯誤)"
        token_losses_np = None # 預設為 None

        if token_count <= 1:
            prediction_text = "⚠️ 警告：輸入文本過短，無法計算有效的困惑度。"
        else:
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                overall_loss = outputs.loss.item()
                logits = outputs.logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                token_losses_np = token_losses.detach().cpu().numpy()

                avg_ppl = np.exp(overall_loss)
                var_token_losses = np.var(token_losses_np)

            # 判斷邏輯不變
            ai_ai_threshold = 30       
            ai_mix_threshold = 100       
            ai_var_loss_threshold = 13 

            if avg_ppl < ai_ai_threshold and var_token_losses < ai_var_loss_threshold:
                prediction_text = "🤖 極高可能是AI生成內容 (PPL極低，高度可預測且平滑)"
            elif avg_ppl < ai_ai_threshold and var_token_losses >= ai_var_loss_threshold:
                prediction_text = "🤖 可能是AI生成，但包含非典型模式 (PPL低，但詞語預測難度波動較大)"
            elif avg_ppl >= ai_ai_threshold and avg_ppl < ai_mix_threshold and var_token_losses < ai_var_loss_threshold:
                prediction_text = "🤔 可能是AI生成或經過高度潤飾的內容 (PPL中等，但結構極為平穩)"
            elif avg_ppl >= ai_ai_threshold and avg_ppl < ai_mix_threshold and var_token_losses >= ai_var_loss_threshold:
                prediction_text = "✅ 較可能是人類撰寫 (PPL中等，語氣或表達具備自然波動)"
            else: 
                prediction_text = "✅ 極高可能是人類撰寫 (PPL高，模型預測困難，符合人類寫作特點)"
        
        # 返回所有需要的數據，包括用於高亮的 offsets 和 token_losses_np
        return avg_ppl, var_token_losses, token_count, prediction_text, offsets, token_losses_np

if __name__ == "__main__":
    root = tk.Tk()
    app = PerplexityApp(root)
    root.mainloop()