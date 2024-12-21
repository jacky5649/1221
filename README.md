# Kaggle首戰斬獲第三，看深度學習菜鳥團隊如何一鳴驚人。
南華大學跨領域-人工智慧期中報告
11223025林彥融
# 從一個簡單易懂的深度學習指南開始

不要想著馬上就能理解所有東西，這需要大量的練習。本指南旨在向深度學習初學者展示 fast.ai 的魅力。假定你了解一些 python 知識，也對機器學習稍有涉獵。這樣的話，我們就走上了學習正軌。

（引用）本文展示的所有程式碼可在 Google Colaboratory 中找到：這是一個 Jupyter 筆記本環境，不需要進行任何設定就可以使用，並且完全在雲端運行。你可以透過 Colaboratory 編寫和執行程式碼，保存和分享分析，存取大量的運算資源，所有這些都是免費的。

程式碼請參閱：https://colab.research.google.com/drive/1PVaRPY1XZuPLtm01V2XxIWqhLrz3_rgX。

# 導入 fast.ai 和我們將要使用的其他函式庫

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_123317_www.sohu.com.jpeg)

輸入庫

# 取得競賽數據

為了盡可能簡潔明了，Abdishakur 上傳競賽資料檔至 dropbox.com。你可以在競賽頁面上找到這些數據。你需要接受競賽規則，並在參賽後存取資料。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_123747_www.sohu.com.jpeg)

# 觀察數據

我們在解決一個問題時首先要做的是觀察可用資料。在想出解決方案之前，我們需要先理解這個問題以及資料是什麼樣的。觀察資料意味著理解資料目錄的構成方式、資料標籤以及樣本影像是什麼樣的。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_124210_www.sohu.com.jpeg)

使用 pandas 庫來讀取資料。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_124438_www.sohu.com.jpeg)

訓練模型所要使用的資料標籤。


處理「影像分類資料集」和「表格資料集」的主要區別在於標籤的儲存方式。這裡的標籤指的是圖像中的內容。在這個特定的資料集中，標籤以 CSV 檔案格式儲存。

想要了解更多計算「分數」欄位的方法，點選：

https://success.figure-eight.com/hc/en-us/articles/201855939-How-to-Calculate-a-Confidence-Score。

我們將使用 seaborn 的 countplot 函數來觀察訓練資料的分佈。我們從下圖中看到，大約 14300 個圖像中沒有發現油棕種植園，而僅有 942 個圖像中發現了油棕種植園。這就是所謂的不平衡資料集，但我們在這裡不討論這個深度學習問題。我們此刻正邁出了一小步。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_124753_www.sohu.com.jpeg)

統計兩個類別的樣本數。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_124955_www.sohu.com.jpeg)

訓練資料集中分佈

#準備數據

提供的測試資料放置於兩個不同的資料夾：leaderboard 留出資料和 leaderboard 測試資料。由於競賽要求提交這兩種資料集的預測，所以我們將兩者結合。我們共獲得 6534 副圖像。

![image](


