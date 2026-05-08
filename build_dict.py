import json
import re
from collections import Counter, defaultdict

# 这是我们要生成的轻量级分诊字典
symptom_to_dept = defaultdict(list)
success_count = 0

print("🔥 开始启动数据炼丹炉，提炼千万级医疗知识...")

try:
    with open('./medical.json', 'r', encoding='utf-8') as f:
        # 一次性把整个几十 MB 的文件读取进内存
        content = f.read().strip()
        
        print("⏳ 正在对非标准 JSON 数据进行智能清洗...")
        # 核心黑科技：修复 MongoDB 导出的非标准格式
        # 它的数据全是独立的 {} {}，我们在中间加上逗号，并在最外层包上中括号
        if content.startswith('{') and content.endswith('}'):
            content = re.sub(r'\}\s*\{', '},{', content)
            content = '[' + content + ']'
            
        # 现在把它作为标准的 Python 列表去解析
        data_list = json.loads(content)
        
        for data in data_list:
            # 确保当前条目是个字典
            if isinstance(data, dict):
                symptoms = data.get('symptom', [])
                depts = data.get('cure_department', [])
                
                # 建立 症状 -> 科室 的映射
                if symptoms and depts:
                    target_dept = depts[-1] 
                    for sym in symptoms:
                        symptom_to_dept[sym].append(target_dept)
                success_count += 1
                
    print(f"✅ 成功读取了 {success_count} 种疾病的知识！")
    
    # 统计最高频科室
    final_triage_dict = {}
    for sym, dept_list in symptom_to_dept.items():
        most_common_dept = Counter(dept_list).most_common(1)[0][0]
        final_triage_dict[sym] = f"急诊{most_common_dept}"

    # 导出精简版字典
    with open('./triage_dict.json', 'w', encoding='utf-8') as f:
        json.dump(final_triage_dict, f, ensure_ascii=False, indent=2)
        
    print(f"🎉 炼丹完成！成功提炼出包含 {len(final_triage_dict)} 个症状的专属分诊字典：triage_dict.json")

except Exception as e:
    print(f"❌ 发生错误：{e}")