import os
print(os.getcwd())
import sys
sys.path.append('/Project/Yi/defense')
from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Model.inception import *
from CODE.Utils.augmentation import *

results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for dataset_name in UNIVARIATE_DATASET_NAMES:
    start_time = time.time()  # 记录训练开始时间
    train_loader, test_loader, train_shape, test_shape, ucr_nb_classes = ucr_loader(dataset_name, batch_size=256)
    
    model = Classifier_INCEPTION(input_shape=train_shape, nb_classes=ucr_nb_classes).to(device)
    # 选择损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1000):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    torch.save(model.state_dict(), f"/Project/Yi/defense/OUTPUT/0trainingtime/{dataset_name}_0_model_weights.pth")
    #model.defense.print_method_usage()
    end_time = time.time()  # 记录训练结束时间
    total_time = end_time - start_time  # 计算总训练时间
    # 重新创建模型实例并加载权重

    model = Classifier_INCEPTION(input_shape=train_shape, nb_classes=ucr_nb_classes).to(device)
    model.load_state_dict(torch.load(f"/Project/Yi/defense/OUTPUT/0trainingtime/{dataset_name}_0_model_weights.pth"))
    
    model.eval()  # 将模型设置为评估模式
    dataset_results = []  # 初始化一个空列表来保存当前数据集的结果
    for i in range(5):
        
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # 将标签和预测添加到列表中
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # 计算准确率和F1分数
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        dataset_results.append({
            'Accuracy': accuracy * 100,  # 转换为百分比形式
            'F1 Score': f1
        })

    # 计算当前数据集的平均性能指标
    dataset_results_df = pd.DataFrame(dataset_results)
    mean_accuracy = dataset_results_df['Accuracy'].mean()
    mean_f1 = dataset_results_df['F1 Score'].mean()

    # 将平均结果添加到总结果列表中
    results.append({
        'Dataset': dataset_name,
        'Average Accuracy': mean_accuracy,
        'Average F1 Score': mean_f1,
        'Total Training Time': total_time
    })

# 将总结果列表转换为 DataFrame 并打印
results_df = pd.DataFrame(results)
# 定义 CSV 文件路径
csv_file_path = '/Project/Yi/defense/OUTPUT/0trainingtime/results0new.csv'

# 将 DataFrame 保存到 CSV 文件
# 如果文件不存在，就创建一个新文件并写入，如果文件已存在，就追加到文件中
if os.path.exists(csv_file_path):
    results_df.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(csv_file_path, mode='w', header=True, index=False)

print("Results have been saved to CSV.")