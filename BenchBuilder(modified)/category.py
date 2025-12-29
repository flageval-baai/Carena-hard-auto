import ast
import re


class Category:
    def __init__(self):
        pass

    @staticmethod
    def create_category(name):
        if name == "criteria_v0.1":
            return CategoryHardPrompt()
        raise Exception(f"Category name is incorrect: {name}")

    def post_process(self):
        pass


class CategoryHardPrompt(Category):
    def __init__(self):
        super().__init__()
        self.name_tag = "criteria_v0.1"
        self.pattern = re.compile(r"\[[\s]*([1234567][\s]*(?:,[\s]*[1234567][\s]*)*)\]")
        self.sys_prompt = """你的任务是评估以下输入提示对高级人工智能助手能力的评估效果。

对于输入提示，请根据以下 7 个标准进行分析。
1. 具体性：提示是否要求明确的输出形式，例如具体的建议、解决方案、分析结果或操作步骤？这种具体性能够测试AI理解用户真实意图并生成针对性回答的能力。
2. 常识可理解性：提示所涉及的概念、背景知识是否在普通网民的认知范围内，不需要专业领域的深度知识？这有助于评估AI与大众用户沟通的适应性。
3. 复杂性：提示的思考层次是否多样化，从简单的信息查询到需要综合分析、多步推理的复合问题？这可以测试AI处理不同认知负荷任务的能力。
4. 问题解决导向：提示是否明确要求AI提供解决方案或实用建议，而不仅仅是信息罗列或知识复述？这考验AI的实际问题分析和方案生成能力。
5. 创造性需求：提示是否需要AI提供多样化、创新性的思路或表达方式，而不是标准化的答案？这能够评估AI的思维灵活性和表达多样性。
6. 表达准确性：提示本身的语言表述是否清晰准确，没有歧义或表达错误？准确的提示才能让AI准确理解并给出相应的高质量回答。
7. 现实相关性：提示是否基于真实的生活场景和实际需求，而非脱离现实的假想问题？这确保评测能反映AI在真实应用中的表现。

您必须以Python数组的格式列出题目满足的标准编号。例如，"[...]"。不要解释你的选择。"""
        self.tags = {
            1: "具体性",
            2: "常识可理解性",
            3: "复杂性",
            4: "问题解决导向",
            5: "创造性需求",
            6: "表达准确性",
            7: "现实相关性",
        }

    def get_score(self, judgment):
        matches = self.pattern.findall(judgment)
        
        if not matches:
            print(f"No match found in judgment: {judgment[:200]}...")
            return []
        
        match_str = matches[0]
        
        try:
            clean_str = re.sub(r'\s+', '', match_str)
            numbers = [int(x) for x in clean_str.split(',') if x.strip().isdigit()]
            valid_numbers = [n for n in numbers if 1 <= n <= 7]
            return valid_numbers
        except Exception as e:
            print(f"Error parsing judgment: {e}, original: {match_str}")
            return []

    def pre_process(self, prompt):
        conv = [{"role": "system", "content": self.sys_prompt}]
        conv.append({"role": "user", "content": prompt})
        return conv

    def post_process(self, judgment):
        criteria = self.get_score(judgment=judgment)
        
        print(f"Parsed criteria: {criteria}")
        
        if not criteria:
            return {name: False for i, name in self.tags.items()}
        
        return {name: bool(i in criteria) for i, name in self.tags.items()}