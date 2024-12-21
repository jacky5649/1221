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

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_125333_www.sohu.com.jpeg)

結合 leaderboard 留出資料和 leaderboard 測試資料。

我們將使用 fast.ai 的 DataBlock API 來構成數據，這是將資料集呈現給模型的簡單方法。

![imagr](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_125522_www.sohu.com.jpeg)

建立一個 ImageList 來保留數據

我們將使用 ImageList 來保存訓練數據，並使用 from_df 方法讀取資料。這樣做的原因是，我們將訓練集資訊儲存在了名為 df 的 DataFrame 中。

接下來需要隨機分割訓練集，並保留 20% 作為驗證集，以便在訓練中監督模型表現。我們選擇了一個 seed，確保再一次訓練時能得到相同的結果，透過相同的 seed，我們就能知道哪些改進是好的，哪些是壞的。

此外，我們同樣還要把訓練集的標籤位址提供給 ImageList，並將資料與標籤合併。

最後，還需要在資料上執行轉換，透過設定 flip_vert = True 將翻轉影像，這能幫助模型辨識不同朝向的影像。此外，還需要使用 imagenet_stats 來歸一化映像。


# 預覽影像

如下是有或沒有油棕種植園的衛星圖：

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_131638_www.sohu.com.jpeg)

展示兩個 Batch 的圖像。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_131827_www.sohu.com.jpeg)

有油棕的圖像標記為 1，沒有油棕的標記為 0

# 訓練我們的模型

現在，開始訓練我們的模型。我們將使用卷積神經網路作為主體，並利用 ResNet 模型的預訓練權重。 ResNet 模型被訓練用來對各種影像進行分類，不用擔心它的理論和實作細節。現在，我們建立的模型以衛星圖像作為輸入，並輸出這兩個類別的預測機率。

卷積神經網絡

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_132244_www.sohu.com.jpeg)

搜尋最佳模型學習率。

接下來，我們用 lr_find() 函數找到了理想的學習率，並使用 recorder.plot() 進行了視覺化。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_132851_www.sohu.com.jpeg)

搜尋最佳模型學習率。

我們將選擇斜率最大的學習率，在這裡我們選擇的是 1e-2。




