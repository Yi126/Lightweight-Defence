import os
print(os.getcwd())
import sys
sys.path.append('/Project/Yi/defense')
from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Model.inception import *
from CODE.Utils.augmentation import *

def distillation_loss(output, labels, teacher_output, T=10.0, alpha=0.5):
    hard_loss = nn.CrossEntropyLoss()(output, labels)
    soft_loss = nn.KLDivLoss()(F.log_softmax(output/T, dim=1),
                               F.softmax(teacher_output/T, dim=1))
    return alpha * hard_loss + (1 - alpha) * soft_loss

def train_model(model, train_loader, criterion, optimizer, T=10.0, is_student=False, teacher_model=None, device=None):
    
    model.to(device)  # 确保模型在正确的设备上
    for epoch in range(1000):
        if teacher_model:
            teacher_model.to(device)  # 确保教师模型也在同一设备上
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if is_student and teacher_model:
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                loss = distillation_loss(outputs, labels, teacher_outputs)
            else:
                outputs = outputs / T  # Adjust logits by temperature before loss computation

                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    return model

results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for dataset_name in UNIVARIATE_DATASET_NAMES:
    start_time = time.time()  # 记录训练开始时间
    train_loader, test_loader, train_shape, test_shape, ucr_nb_classes = ucr_loader(dataset_name, batch_size=128)
    
    teacher_model = Classifier_INCEPTION_Logits(input_shape=train_shape, nb_classes=ucr_nb_classes).to(device)
    student_model = Classifier_INCEPTION_Logits(input_shape=train_shape, nb_classes=ucr_nb_classes).to(device)
    # 选择损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    teacher_optimizer = optim.Adam(teacher_model.parameters())
    student_optimizer = optim.Adam(student_model.parameters())

    #model.train()
    '''
    for inputs, labels in train_loader:
        
        teacher_model.train
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = teacher_model(inputs) #need to save the outputs of teacher
        loss = criterion(outputs, labels)
        '''
    # Train teacher model
    teacher_model = train_model(teacher_model, train_loader, criterion, teacher_optimizer, device=device)

    # Train student model with knowledge distillation
    

    student_model = train_model(student_model, train_loader, criterion, student_optimizer, is_student=True, teacher_model=teacher_model, device=device)
    
    #print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(student_model.state_dict(), f"/Project/Yi/defense/OUTPUT/kd/{dataset_name}_kd_model_weights.pth")
    end_time = time.time()  # 记录训练结束时间
    total_time = end_time - start_time  # 计算总训练时间
    # 重新创建模型实例并加载权重

    student_model = Classifier_INCEPTION(input_shape=train_shape, nb_classes=ucr_nb_classes).to(device)
    student_model.load_state_dict(torch.load(f"/Project/Yi/defense/OUTPUT/kd/{dataset_name}_kd_model_weights.pth"))
    # Testing student model
    student_model.eval()
    dataset_results = []  # 初始化一个空列表来保存当前数据集的结果
    for i in range(5):
        
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student_model(inputs)
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
csv_file_path = '/Project/Yi/defense/OUTPUT/kd/resultskd.csv'

# 将 DataFrame 保存到 CSV 文件
# 如果文件不存在，就创建一个新文件并写入，如果文件已存在，就追加到文件中
if os.path.exists(csv_file_path):
    results_df.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(csv_file_path, mode='w', header=True, index=False)

print("Results have been saved to CSV.")