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

class PerplexityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPT-2 文章困惑度分析")
        self.root.geometry("800x700") # 稍微加大視窗以容納新元件

        self.model = None
        self.tokenizer = None
        
        # 預設使用最小的模型
        self.current_model_name = MODEL_OPTIONS["GPT-2 Small (117M)"] 

        self._create_widgets()
        self._load_model_async()

    # --- 新增功能：清空文本 ---
    def _clear_input(self):
        """清空輸入文本框的內容"""
        self.input_text.delete("1.0", tk.END)

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="15 15 15 15")
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)

        # 1. 模型選擇和狀態區域
        config_frame = ttk.Frame(main_frame)
        config_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)
        
        ttk.Label(config_frame, text="選擇模型大小:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        # 模型選擇下拉菜單
        self.model_var = tk.StringVar(value="GPT-2 Small (117M)")
        self.model_combobox = ttk.Combobox(
            config_frame, 
            textvariable=self.model_var, 
            values=list(MODEL_OPTIONS.keys()),
            state="readonly",
            width=25
        )
        self.model_combobox.grid(row=0, column=1, sticky="w", padx=(0, 10))
        # 綁定模型變更事件
        self.model_combobox.bind("<<ComboboxSelected>>", self._on_model_select_change)

        # 載入狀態標籤
        self.status_var = tk.StringVar()
        self.status_var.set("正在載入模型，請稍候...")
        self.status_label = ttk.Label(config_frame, textvariable=self.status_var, font=('Arial', 10, 'italic'), foreground='blue')
        self.status_label.grid(row=0, column=2, sticky="e")
        config_frame.columnconfigure(2, weight=1) # 讓狀態標籤推到最右邊

        # 模型加載進度條 (indeterminate)
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate', length=200)
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(0, 10))


        # 2. 輸入文本區域
        input_controls_frame = ttk.Frame(main_frame)
        input_controls_frame.grid(row=2, column=0, sticky="ew")
        input_controls_frame.columnconfigure(0, weight=1) # 讓標籤靠左
        
        ttk.Label(input_controls_frame, text="請輸入您的英文文章：", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        # 新增清空按鈕
        clear_button = ttk.Button(input_controls_frame, text="清空文本", command=self._clear_input)
        clear_button.grid(row=0, column=1, sticky="e", padx=(10, 0))

        self.input_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=80, height=15, font=('Arial', 10))
        self.input_text.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        
        self.input_text.bind("<Control-a>", self.select_all_text)
        self.input_text.bind("<Command-a>", self.select_all_text) 

        # 3. 計算按鈕
        self.calculate_button = ttk.Button(main_frame, text="計算困惑度", command=self._start_calculation, state=tk.DISABLED)
        self.calculate_button.grid(row=4, column=0, sticky="ew", pady=(0, 20))

        # 4. 結果顯示區域
        ttk.Label(main_frame, text="=== 分析結果 ===", font=('Arial', 12, 'bold')).grid(row=5, column=0, sticky="w", pady=(10, 5))

        # 結果顯示的子框架，方便使用 grid 進行對齊
        result_frame = ttk.Frame(main_frame)
        result_frame.grid(row=6, column=0, sticky="ew", padx=10)
        result_frame.columnconfigure(0, weight=1) # 左邊文字可以擴展

        # Token 數量 (新增)
        self.token_count_var = tk.StringVar(value="")
        ttk.Label(result_frame, text="分析的 Token 總數：", font=('Arial', 11)).grid(row=0, column=0, sticky="w")
        ttk.Label(result_frame, textvariable=self.token_count_var, font=('Arial', 11, 'bold'), foreground='darkcyan').grid(row=0, column=1, sticky="e")
        
        # 困惑度
        self.ppl_var = tk.StringVar(value="")
        ttk.Label(result_frame, text="整體平均困惑度（PPL）：", font=('Arial', 11)).grid(row=1, column=0, sticky="w")
        ttk.Label(result_frame, textvariable=self.ppl_var, font=('Arial', 11, 'bold'), foreground='darkgreen').grid(row=1, column=1, sticky="e")

        # Token 損失變異量
        self.var_loss_var = tk.StringVar(value="")
        ttk.Label(result_frame, text="Token 損失變異量：", font=('Arial', 11)).grid(row=2, column=0, sticky="w")
        ttk.Label(result_frame, textvariable=self.var_loss_var, font=('Arial', 11, 'bold'), foreground='darkgreen').grid(row=2, column=1, sticky="e")

        # 分隔線
        ttk.Separator(main_frame, orient='horizontal').grid(row=7, column=0, sticky="ew", pady=(10, 10))

        # 判斷結果
        self.prediction_var = tk.StringVar(value="")
        ttk.Label(main_frame, text="判斷結果：", font=('Arial', 12)).grid(row=8, column=0, sticky="w", padx=(10, 0))
        self.prediction_label = ttk.Label(main_frame, textvariable=self.prediction_var, font=('Arial', 14, 'bold'))
        self.prediction_label.grid(row=8, column=0, sticky="e", padx=(10, 10))

        # 底部說明
        ttk.Label(main_frame, text="提示：PPL 越低，通常表示文本對模型而言越容易預測。", font=('Arial', 9, 'italic')).grid(row=9, column=0, sticky="w", pady=(10, 0))
        ttk.Label(main_frame, text="注意：這些判斷閾值是經驗性的，可能需要根據實際應用調整。", font=('Arial', 9, 'italic')).grid(row=10, column=0, sticky="w", pady=(0, 5))

        # 設置行和列的權重
        main_frame.rowconfigure(3, weight=1) # 輸入文本區域可以擴展

    def select_all_text(self, event=None):
        self.input_text.tag_add("sel", "1.0", "end-1c")
        return "break" 

    # --- 新增功能：處理模型選擇變更 ---
    def _on_model_select_change(self, event):
        """當模型下拉菜單改變時，觸發重新載入模型"""
        selected_key = self.model_var.get()
        new_model_name = MODEL_OPTIONS.get(selected_key)
        
        if new_model_name != self.current_model_name:
            self.current_model_name = new_model_name
            self._load_model_async()

    def _load_model_async(self):
        """在單獨的執行緒中加載模型，避免阻塞 GUI"""
        self.calculate_button.config(state=tk.DISABLED)
        # 顯示正在載入的模型名稱
        self.status_var.set(f"正在載入 {self.current_model_name} 模型... 請稍候...")
        self.progress_bar.start(10) # 啟動不確定模式進度條
        self.root.update_idletasks() 

        def load_task():
            try:
                # 確保使用當前選擇的模型名稱
                model = GPT2LMHeadModel.from_pretrained(self.current_model_name)
                tokenizer = GPT2TokenizerFast.from_pretrained(self.current_model_name)
                model.eval()
                
                # 更新實例變數
                self.model = model
                self.tokenizer = tokenizer
                
                self.root.after(0, self._on_model_loaded, True) 
            except Exception as e:
                self.root.after(0, self._on_model_loaded, False, str(e))

        threading.Thread(target=load_task).start()

    def _on_model_loaded(self, success, error_message=None):
        self.progress_bar.stop() # 停止進度條
        if success:
            self.status_var.set(f"模型 {self.current_model_name} 載入完成。")
            self.calculate_button.config(state=tk.NORMAL)
        else:
            self.status_var.set(f"模型載入失敗: {error_message}")
            messagebox.showerror("錯誤", f"無法載入模型：{error_message}\n請檢查網絡連接或模型名稱。")
            self.calculate_button.config(state=tk.DISABLED)

    def _start_calculation(self):
        """啟動計算，並在單獨的執行緒中執行"""
        text = self.input_text.get("1.0", "end-1c").strip() 

        if not text:
            messagebox.showwarning("輸入錯誤", "請在文本框中輸入文章後再計算。")
            return

        if self.model is None or self.tokenizer is None:
            messagebox.showerror("錯誤", "模型尚未載入完成，請稍候。")
            return

        self.calculate_button.config(state=tk.DISABLED)
        self.status_var.set("正在計算中...請稍候...")
        self.ppl_var.set("")
        self.var_loss_var.set("")
        self.token_count_var.set("") # 清空 Token Count
        self.prediction_var.set("")
        self.progress_bar.start(50) # 啟動計算進度條
        self.root.update_idletasks() 

        def calculation_task():
            try:
                avg_ppl, var_token_losses, token_count, prediction_text = self._calculate_perplexity(text)
                self.root.after(0, self._on_calculation_complete, avg_ppl, var_token_losses, token_count, prediction_text)
            except Exception as e:
                self.root.after(0, self._on_calculation_error, str(e))

        threading.Thread(target=calculation_task).start()

    def _on_calculation_complete(self, avg_ppl, var_token_losses, token_count, prediction_text):
        """計算完成後更新 GUI"""
        self.progress_bar.stop() # 停止進度條
        
        # 顯示 Token 數量 (新增)
        self.token_count_var.set(f"{token_count}")
        
        self.ppl_var.set(f"{avg_ppl:.2f}" if avg_ppl != float('inf') else "N/A (文本過短)")
        self.var_loss_var.set(f"{var_token_losses:.2f}" if var_token_losses != float('inf') else "N/A (文本過短)")
        self.prediction_var.set(prediction_text)
        self.status_var.set("計算完成。")
        self.calculate_button.config(state=tk.NORMAL)

        # 根據預測結果調整預測文字的顏色
        if "極高可能是AI生成內容" in prediction_text:
            self.prediction_label.config(foreground='red')
        elif "可能是AI生成" in prediction_text:
            self.prediction_label.config(foreground='orange')
        elif "較可能是人類撰寫" in prediction_text:
            self.prediction_label.config(foreground='darkblue')
        elif "極高可能是人類撰寫" in prediction_text:
            self.prediction_label.config(foreground='green')
        else:
            self.prediction_label.config(foreground='purple') 

    def _on_calculation_error(self, error_message):
        """計算出錯時更新 GUI"""
        self.progress_bar.stop() # 停止進度條
        self.status_var.set("計算失敗。")
        messagebox.showerror("錯誤", f"計算過程中發生錯誤：{error_message}")
        self.calculate_button.config(state=tk.NORMAL)


    def _calculate_perplexity(self, text):
        """
        核心困惑度計算邏輯，並返回 Token 數量 (新增)
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        token_count = input_ids.shape[1] # 獲取 Token 數量
        avg_ppl = float('inf')
        var_token_losses = float('inf')
        prediction_text = "無法判斷 (文本過短或錯誤)"

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

            # 簡單判斷（可微調閾值）
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
        
        return avg_ppl, var_token_losses, token_count, prediction_text # 返回 Token 數量

if __name__ == "__main__":
    root = tk.Tk()
    app = PerplexityApp(root)
    root.mainloop()