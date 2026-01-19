import ast
import random
import json
import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import copy

# =========================================================
# Sam導師核心思想整合系統
# =========================================================

class SamMetaValidator:
    """
    Sam導師元驗證器 - 檢查邏輯是否符合"進化就是先模仿邏輯方式，運行邏輯觀測，變異邏輯分支，創造新邏輯演繹"
    """
    
    @staticmethod
    def validate_evolution_logic(inference) -> Tuple[bool, str]:
        """
        驗證是否體現Sam導師的進化思想
        返回：(是否通過, 反饋信息)
        """
        feedback = []
        
        # 檢查是否有模仿階段
        if hasattr(inference, 'generation') and inference.generation == 0:
            feedback.append("✅ 包含初始模仿結構")
        
        # 檢查是否有變異歷史
        if hasattr(inference, 'premises'):
            for p in inference.premises:
                if hasattr(p, 'mutation_history') and p.mutation_history:
                    feedback.append("✅ 包含邏輯變異歷史")
                    break
        
        # 檢查是否有新邏輯創造
        if hasattr(inference, 'novelty_score') and inference.novelty_score > 0.5:
            feedback.append("✅ 包含新邏輯創造")
        
        # 檢查是否有觀測(自省)
        if hasattr(inference, 'self_observation'):
            feedback.append("✅ 包含邏輯自我觀測")
        
        return len(feedback) > 0, " | ".join(feedback) if feedback else "未體現進化邏輯"

# =========================================================
# 增強型基礎結構
# =========================================================

class EnhancedPremise(Premise):
    def __init__(self, text: str):
        super().__init__(text)
        self.logical_form = self._analyze_logical_form(text)
        self.entropy = random.random()  # 變異潛力
        self.self_contradiction = False
        self.evolution_path = []
    
    def _analyze_logical_form(self, text: str) -> Dict[str, Any]:
        """深度分析邏輯形式"""
        return {
            "subject": self._extract_subject(text),
            "predicate": self._extract_predicate(text),
            "quantifier": self._detect_quantifier(text),
            "modality": self._detect_modality(text),
            "truth_value": self._estimate_truth_value(text),
            "complexity": len(text.split()) / 10.0
        }
    
    def _extract_subject(self, text: str) -> str:
        """提取主語"""
        words = text.split()
        if "是" in text:
            parts = text.split("是")
            return parts[0].strip() if len(parts) > 1 else text
        return words[0] if words else ""
    
    def _extract_predicate(self, text: str) -> str:
        """提取謂語"""
        if "是" in text:
            parts = text.split("是")
            return parts[1].strip() if len(parts) > 1 else ""
        return text
    
    def _detect_quantifier(self, text: str) -> str:
        """檢測量詞"""
        quantifiers = ["所有", "每個", "任何", "有些", "存在", "部分", "大多數"]
        for q in quantifiers:
            if q in text:
                return q
        return "無"
    
    def _detect_modality(self, text: str) -> str:
        """檢測模態"""
        modalities = ["必須", "應該", "可能", "可以", "必然", "不可能"]
        for m in modalities:
            if m in text:
                return m
        return "斷言"
    
    def _estimate_truth_value(self, text: str) -> float:
        """估計真值（簡化）"""
        negative_words = ["不", "非", "無", "沒有", "假"]
        for word in negative_words:
            if word in text:
                return 0.3
        return 0.7

class QuantumClaim(Claim):
    def __init__(self, text):
        super().__init__(text)
        self.superposition = []
        self.domain = "general"
        self.logical_depth = 0
        self.paradox_tolerance = 0.5
        self.quantum_entanglement = []  # 與其他結論的量子糾纏
        
    def generate_meaningful_superposition(self, premises: List[EnhancedPremise]):
        """
        生成有意義的邏輯疊加態
        基於前提分析產生相關的替代結論
        """
        # 分析前提的共同主題
        subjects = [p.logical_form["subject"] for p in premises if p.logical_form["subject"]]
        predicates = [p.logical_form["predicate"] for p in premises if p.logical_form["predicate"]]
        
        if subjects and predicates:
            common_subject = max(set(subjects), key=subjects.count) if subjects else ""
            common_predicate = max(set(predicates), key=predicates.count) if predicates else ""
            
            # 生成邏輯相關的疊加態
            if common_subject and common_predicate:
                self.superposition.extend([
                    f"相反地，{common_subject}可能不是{common_predicate}",
                    f"在某些條件下，{common_subject}是{common_predicate}的對立面",
                    f"{common_subject}與{common_predicate}的關係是辯證的"
                ])
        
        # 添加邏輯學標準疊加態
        self.superposition.extend([
            f"從另一角度，{self.text}",
            f"考慮邊界情況，{self.text}可能不成立",
            f"{self.text}的反命題也值得考慮"
        ])
        
        self.logical_depth = len(self.superposition) / 10.0

# =========================================================
# 增強型推理結構
# =========================================================

class EvolutionaryInference(Inference):
    def __init__(self, premises: List[EnhancedPremise], claim: QuantumClaim):
        # 確保使用增強型前提
        enhanced_premises = []
        for p in premises:
            if not isinstance(p, EnhancedPremise):
                enhanced_premises.append(EnhancedPremise(p.text))
            else:
                enhanced_premises.append(p)
        
        # 生成有意義的量子疊加態
        if not claim.superposition:
            claim.generate_meaningful_superposition(enhanced_premises)
        
        super().__init__(enhanced_premises, claim)
        self.generation = 0
        self.fitness_scores = {}
        self.total_fitness = 0.0
        self.novelty_score = 0.0
        self.sam_alignment_score = 0.0
        self.creation_time = datetime.now().isoformat()
        self.domain = claim.domain
        self.logical_consistency = 0.0
        self.self_observation = []  # 自我觀測記錄
        self.evolution_feedback = ""  # 進化反饋
        
        # 計算邏輯一致性
        self.calculate_logical_consistency()
    
    def calculate_logical_consistency(self):
        """計算前提間的邏輯一致性"""
        if len(self.premises) < 2:
            self.logical_consistency = 1.0
            return
        
        consistency_scores = []
        for i in range(len(self.premises)):
            for j in range(i+1, len(self.premises)):
                p1 = self.premises[i]
                p2 = self.premises[j]
                
                # 檢查邏輯衝突
                conflict = self._check_premise_conflict(p1, p2)
                consistency_scores.append(1.0 - conflict)
        
        self.logical_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _check_premise_conflict(self, p1: EnhancedPremise, p2: EnhancedPremise) -> float:
        """檢查兩個前提的衝突程度"""
        # 簡單的衝突檢測
        if p1.is_affirmative != p2.is_affirmative:
            return 0.3
        
        # 檢查主語-謂語矛盾
        if p1.logical_form["subject"] == p2.logical_form["subject"]:
            if p1.logical_form["predicate"] != p2.logical_form["predicate"]:
                return 0.5
        
        return 0.0
    
    def mutate_with_sam_logic(self):
        """
        按照Sam導師的進化邏輯進行變異：
        1. 模仿 → 2. 觀測 → 3. 變異 → 4. 創造
        """
        # 1. 模仿階段：記錄當前狀態
        original_state = self._capture_state()
        self.self_observation.append({
            "stage": "模仿",
            "state": original_state,
            "timestamp": datetime.now().isoformat()
        })
        
        # 2. 觀測階段：分析邏輯結構
        analysis = self._analyze_logic_structure()
        self.self_observation.append({
            "stage": "觀測",
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })
        
        # 3. 變異階段：智能變異
        mutated = self._intelligent_mutation(analysis)
        
        # 4. 創造階段：產生新邏輯
        if mutated:
            new_logic = self._create_new_logic_pattern()
            mutated.claim.text = new_logic if random.random() > 0.5 else mutated.claim.text
        
        return mutated if mutated else self
    
    def _capture_state(self) -> Dict:
        """捕獲當前狀態"""
        return {
            "premises": [p.text for p in self.premises],
            "claim": self.claim.text,
            "fitness": self.total_fitness,
            "consistency": self.logical_consistency
        }
    
    def _analyze_logic_structure(self) -> Dict:
        """分析邏輯結構"""
        structure = {
            "premise_count": len(self.premises),
            "premise_types": {
                "universal": sum(1 for p in self.premises if p.is_universal),
                "affirmative": sum(1 for p in self.premises if p.is_affirmative),
                "complex": sum(1 for p in self.premises if p.logical_form["complexity"] > 0.5)
            },
            "inference_pattern": self._detect_inference_pattern(),
            "weak_points": self._identify_weak_points()
        }
        return structure
    
    def _detect_inference_pattern(self) -> str:
        """檢測推理模式"""
        patterns = {
            "deductive": ["所有", "都", "必然"],
            "inductive": ["大多數", "通常", "往往"],
            "abductive": ["可能", "推測", "假設"]
        }
        
        text = " ".join([p.text for p in self.premises]) + " " + self.claim.text
        
        for pattern, keywords in patterns.items():
            for keyword in keywords:
                if keyword in text:
                    return pattern
        
        return "unknown"
    
    def _identify_weak_points(self) -> List[str]:
        """識別邏輯弱點"""
        weak_points = []
        
        # 檢查前提數量
        if len(self.premises) < 2:
            weak_points.append("前提不足")
        
        # 檢查邏輯一致性
        if self.logical_consistency < 0.7:
            weak_points.append(f"邏輯一致性低({self.logical_consistency:.2f})")
        
        # 檢查前提類型多樣性
        universal_count = sum(1 for p in self.premises if p.is_universal)
        if universal_count == len(self.premises):
            weak_points.append("全稱前提過多，缺乏特例")
        
        return weak_points
    
    def _intelligent_mutation(self, analysis: Dict) -> 'EvolutionaryInference':
        """基於分析的智能變異"""
        mutated = copy.deepcopy(self)
        
        # 根據弱點進行變異
        weak_points = analysis.get("weak_points", [])
        
        if "前提不足" in weak_points:
            # 添加新前提
            new_premise_text = self._generate_new_premise()
            mutated.premises.append(EnhancedPremise(new_premise_text))
        
        if "全稱前提過多" in weak_points:
            # 將全稱前提轉為特稱
            for i, p in enumerate(mutated.premises):
                if p.is_universal and random.random() > 0.5:
                    mutated.premises[i] = EnhancedPremise(
                        p.text.replace("所有", "有些").replace("都", "可能")
                    )
        
        # 隨機變異一個前提
        if mutated.premises:
            idx = random.randint(0, len(mutated.premises)-1)
            mutated.premises[idx] = mutated.premises[idx].mutate_logic_ast()
        
        mutated.generation = self.generation + 1
        return mutated
    
    def _generate_new_premise(self) -> str:
        """生成新前提"""
        templates = [
            "考慮到{context}，{subject}具有{property}",
            "從{perspective}角度，{subject}與{relation}相關",
            "在{condition}條件下，{subject}表現為{behavior}"
        ]
        
        subject = self.premises[0].logical_form["subject"] if self.premises else "事物"
        
        context_options = ["歷史發展", "社會環境", "技術進步", "文化背景"]
        property_options = ["複雜性", "多樣性", "動態性", "不確定性"]
        perspective_options = ["系統論", "辯證法", "實用主義", "建構主義"]
        relation_options = ["整體與部分", "原因與結果", "量變與質變", "必然與偶然"]
        condition_options = ["特定環境", "理想狀態", "邊界情況", "極端條件"]
        behavior_options = ["適應", "演化", "突現", "自組織"]
        
        template = random.choice(templates)
        
        return template.format(
            subject=subject,
            context=random.choice(context_options),
            property=random.choice(property_options),
            perspective=random.choice(perspective_options),
            relation=random.choice(relation_options),
            condition=random.choice(condition_options),
            behavior=random.choice(behavior_options)
        )
    
    def _create_new_logic_pattern(self) -> str:
        """創造新邏輯模式"""
        patterns = [
            "辯證統一：{thesis}與{antithesis}的綜合",
            "遞歸自指：關於{subject}的論述本身構成{subject}的一部分",
            "量子疊加：{subject}同時處於{state1}和{state2}的疊加態",
            "元邏輯跳躍：從{level1}層級躍升到{level2}層級理解{subject}"
        ]
        
        subject = self.premises[0].logical_form["subject"] if self.premises else "現實"
        
        thesis_options = ["肯定", "存在", "確定性", "統一"]
        antithesis_options = ["否定", "非存在", "不確定性", "多樣性"]
        state_options = ["有序", "混沌", "穩定", "演化"]
        level_options = ["現象", "本質", "結構", "功能", "關係", "演化"]
        
        pattern = random.choice(patterns)
        
        return pattern.format(
            subject=subject,
            thesis=random.choice(thesis_options),
            antithesis=random.choice(antithesis_options),
            state1=random.choice(state_options),
            state2=random.choice([s for s in state_options if s != state1]),
            level1=random.choice(level_options),
            level2=random.choice([l for l in level_options if l != level1])
        )
    
    def to_dict(self):
        return {
            "generation": self.generation,
            "domain": self.domain,
            "premises": [p.text for p in self.premises],
            "premises_enhanced": [p.logical_form for p in self.premises],
            "claim": self.claim.text,
            "superposition": self.claim.superposition,
            "sam_alignment": self.sam_alignment_score,
            "total_fitness": self.total_fitness,
            "novelty_score": self.novelty_score,
            "logical_consistency": self.logical_consistency,
            "evolution_feedback": self.evolution_feedback,
            "self_observation_count": len(self.self_observation),
            "creation_time": self.creation_time
        }

# =========================================================
# 增強型LLM評分器
# =========================================================

class EnhancedLLMFitnessEvaluator(LLMFitnessEvaluator):
    """
    增強型評分器，整合Sam導師的進化思想評價
    """
    
    @staticmethod
    def evaluate_with_sam_philosophy(inference: EvolutionaryInference) -> Dict[str, float]:
        # 1. 原有評分
        base_scores = LLMFitnessEvaluator.evaluate(inference)
        
        # 2. Sam導師進化思想評分
        sam_evolution_score = EnhancedLLMFitnessEvaluator._evaluate_sam_evolution(inference)
        
        # 3. 邏輯一致性評分
        consistency_score = inference.logical_consistency
        
        # 4. 綜合評分
        weights = {
            "base": 0.6,
            "sam_evolution": 0.3,
            "consistency": 0.1
        }
        
        # 計算加權分數
        base_total = sum(base_scores.values()) / len(base_scores) if base_scores else 0
        weighted_score = (
            base_total * weights["base"] +
            sam_evolution_score * weights["sam_evolution"] +
            consistency_score * weights["consistency"]
        )
        
        # 更新總適應度
        inference.total_fitness = min(1.0, weighted_score)
        
        # 添加反饋
        if sam_evolution_score > 0.7:
            inference.evolution_feedback = "良好體現Sam導師進化思想"
        elif sam_evolution_score < 0.3:
            inference.evolution_feedback = "需加強進化邏輯結構"
        
        return {
            **base_scores,
            "sam_evolution": sam_evolution_score,
            "consistency": consistency_score,
            "weighted_total": inference.total_fitness
        }
    
    @staticmethod
    def _evaluate_sam_evolution(inference: EvolutionaryInference) -> float:
        """評價是否符合Sam導師的進化思想"""
        score = 0.0
        
        # 檢查是否有模仿階段
        if inference.generation > 0:
            score += 0.2
        
        # 檢查是否有自我觀測
        if hasattr(inference, 'self_observation') and inference.self_observation:
            score += 0.3
        
        # 檢查是否有邏輯變異
        premise_variation = False
        for p in inference.premises:
            if hasattr(p, 'mutation_history') and p.mutation_history:
                premise_variation = True
                break
        
        if premise_variation:
            score += 0.3
        
        # 檢查是否有新邏輯創造
        if inference.novelty_score > 0.6:
            score += 0.2
        
        return score

# =========================================================
# 增強型進化引擎
# =========================================================

class EnhancedEvolutionEngine(UltimateEvolutionEngine):
    """
    增強型進化引擎，整合自我否定驗證和自洽優化
    """
    
    def __init__(self, population: PersistentPopulation):
        super().__init__(population)
        self.self_criticism_log = []
        self.optimization_history = []
    
    def evolve_with_self_criticism(self, generations: int = 10, target_size: int = 50):
        """
        帶有自我批判的進化過程
        """
        print("="*80)
        print("🔍 Sam導師 · 自我否定驗證進化引擎啟動")
        print("="*80)
        
        for gen in range(generations):
            print(f"\n🧬 第 {gen+1}/{generations} 代 · 自我批判進化")
            
            # 第一步：自我批判階段
            self._self_criticism_phase()
            
            # 第二步：評分階段（使用增強評分器）
            with ThreadPoolExecutor(max_workers=10) as exec:
                futures = [exec.submit(EnhancedLLMFitnessEvaluator.evaluate_with_sam_philosophy, ind) 
                          for ind in self.pop.population]
                results = []
                for f in as_completed(futures):
                    results.append(f.result())
            
            # 第三步：自洽優化階段
            self._self_consistency_optimization()
            
            # 第四步：核心價值過濾
            current_pop = [ind for ind in self.pop.population if ind.sam_alignment_score >= 0]
            
            if not current_pop:
                print("⚠️ 種群違背核心價值，注入新種子...")
                self._inject_healthy_seeds()
                current_pop = self.pop.population
            
            # 第五步：排序與選擇
            current_pop.sort(key=lambda x: x.total_fitness, reverse=True)
            
            # 第六步：動態變異與繁殖
            new_gen = self._adaptive_reproduction(current_pop, target_size)
            
            self.pop.population = new_gen
            self.pop.save()
            
            # 顯示進化進度
            self._display_generation_progress(gen, current_pop)
        
        # 最終優化
        self._final_optimization()
        
        return self._get_best_solution()
    
    def _self_criticism_phase(self):
        """自我批判階段：尋找並修復邏輯缺陷"""
        print("   🔍 自我批判分析...")
        
        for i, ind in enumerate(self.pop.population[:10]):  # 只分析前10個
            # 檢查邏輯一致性
            if hasattr(ind, 'logical_consistency') and ind.logical_consistency < 0.6:
                self.self_criticism_log.append({
                    "individual": i,
                    "issue": f"邏輯一致性低: {ind.logical_consistency:.2f}",
                    "timestamp": datetime.now().isoformat()
                })
                
                # 嘗試修復
                self._repair_logical_inconsistency(ind)
            
            # 檢查前提多樣性
            if len(ind.premises) > 0:
                universal_count = sum(1 for p in ind.premises if p.is_universal)
                if universal_count == len(ind.premises):
                    self.self_criticism_log.append({
                        "individual": i,
                        "issue": "前提類型單一（全全稱）",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # 嘗試多樣化
                    self._diversify_premises(ind)
    
    def _repair_logical_inconsistency(self, inference: EvolutionaryInference):
        """修復邏輯不一致性"""
        # 尋找衝突的前提對
        conflicts = []
        for i in range(len(inference.premises)):
            for j in range(i+1, len(inference.premises)):
                conflict_score = inference._check_premise_conflict(
                    inference.premises[i], 
                    inference.premises[j]
                )
                if conflict_score > 0.3:
                    conflicts.append((i, j, conflict_score))
        
        # 修復最嚴重的衝突
        if conflicts:
            conflicts.sort(key=lambda x: x[2], reverse=True)
            i, j, _ = conflicts[0]
            
            # 修改其中一個前提
            if random.random() > 0.5:
                inference.premises[i] = inference.premises[i].mutate_logic_ast()
            else:
                inference.premises[j] = inference.premises[j].mutate_logic_ast()
            
            # 重新計算一致性
            inference.calculate_logical_consistency()
    
    def _diversify_premises(self, inference: EvolutionaryInference):
        """多樣化前提"""
        if len(inference.premises) > 0:
            # 隨機選擇一個全稱前提轉為特稱
            for i, p in enumerate(inference.premises):
                if p.is_universal and random.random() > 0.5:
                    new_text = p.text.replace("所有", "有些").replace("都", "可能")
                    inference.premises[i] = EnhancedPremise(new_text)
                    break
    
    def _self_consistency_optimization(self):
        """自洽優化階段"""
        print("   ⚙️  自洽優化中...")
        
        for ind in self.pop.population:
            # 確保前提與結論相關
            self._optimize_premise_relevance(ind)
            
            # 優化量子疊加態的相關性
            self._optimize_superposition_relevance(ind)
            
            # 記錄優化
            self.optimization_history.append({
                "individual_id": id(ind),
                "optimization": "自洽優化",
                "timestamp": datetime.now().isoformat()
            })
    
    def _optimize_premise_relevance(self, inference: EvolutionaryInference):
        """優化前提與結論的相關性"""
        if not inference.premises or not inference.claim:
            return
        
        claim_subject = inference.claim.text.split()[0] if inference.claim.text else ""
        
        # 檢查前提是否包含結論的主題
        relevant_premises = []
        for p in inference.premises:
            if claim_subject in p.text or p.logical_form["subject"] in claim_subject:
                relevant_premises.append(p)
        
        # 如果相關前提太少，添加相關前提
        if len(relevant_premises) < len(inference.premises) * 0.5:
            new_premise = EnhancedPremise(f"{claim_subject}具有相關屬性")
            inference.premises.append(new_premise)
    
    def _optimize_superposition_relevance(self, inference: EvolutionaryInference):
        """優化量子疊加態的相關性"""
        if hasattr(inference.claim, 'superposition') and inference.claim.superposition:
            # 移除不相關的疊加態
            claim_keywords = set(inference.claim.text.split()[:3])
            relevant_superpositions = []
            
            for sup in inference.claim.superposition:
                sup_keywords = set(sup.split()[:3])
                # 計算關鍵詞重疊度
                overlap = len(claim_keywords.intersection(sup_keywords)) / max(len(claim_keywords), 1)
                if overlap > 0.3:  # 30%關鍵詞重疊
                    relevant_superpositions.append(sup)
            
            inference.claim.superposition = relevant_superpositions
    
    def _adaptive_reproduction(self, current_pop: List, target_size: int) -> List:
        """自適應繁殖策略"""
        new_gen = current_pop[:max(8, target_size // 6)].copy()
        
        while len(new_gen) < target_size:
            parent = random.choice(current_pop[:20])  # 從前20個中選擇
            
            # 根據適應度調整變異策略
            if parent.total_fitness > 0.8:
                # 高分個體：細微變異
                child = parent.mutate_with_sam_logic()
            elif parent.total_fitness < 0.4:
                # 低分個體：大幅度變異
                child = self._radical_mutation(parent)
            else:
                # 中等分數：交叉繁殖
                parent2 = random.choice(current_pop[:20])
                child = parent.crossover(parent2)
            
            # 應用Sam導師的進化邏輯
            if random.random() > 0.5:
                child = child.mutate_with_sam_logic()
            
            new_gen.append(child)
        
        return new_gen
    
    def _radical_mutation(self, parent: EvolutionaryInference) -> EvolutionaryInference:
        """大幅度變異"""
        mutated = copy.deepcopy(parent)
        
        # 變異所有前提
        mutated.premises = [p.mutate_logic_ast() for p in mutated.premises]
        
        # 徹底改變結論
        mutated.claim.text = f"重新思考：{mutated.claim.text}"
        mutated.claim.superposition = []
        mutated.claim.generate_meaningful_superposition(mutated.premises)
        
        mutated.generation = parent.generation + 1
        
        return mutated
    
    def _inject_healthy_seeds(self):
        """注入健康的種子論證"""
        healthy_seeds = [
            "愛是人類的核心價值\n和平是愛的表現\n因此我們追求和平",
            "知識需要驗證\n實驗提供驗證\n因此實驗是獲取真知的重要途徑",
            "進化需要變異\n變異產生多樣性\n多樣性促進適應\n因此進化依賴變異",
            "邏輯需要自洽\n自洽需要驗證\n驗證需要實驗\n因此邏輯最終需要實驗驗證",
            "創造來自模仿\n模仿需要觀察\n觀察導致變異\n變異產生創造\n因此創造是一個進化過程"
        ]
        
        for seed in healthy_seeds:
            inference = DiscourseParser.parse(seed)
            # 轉換為增強型
            enhanced_premises = [EnhancedPremise(p.text) for p in inference.premises]
            enhanced_claim = QuantumClaim(inference.claim.text)
            enhanced_inference = EvolutionaryInference(enhanced_premises, enhanced_claim)
            self.pop.add(enhanced_inference)
    
    def _display_generation_progress(self, gen: int, population: List):
        """顯示進化進度"""
        if population:
            best = population[0]
            worst = population[-1]
            
            print(f"   最佳適應度: {best.total_fitness:.3f} | 最差: {worst.total_fitness:.3f}")
            print(f"   Sam對齊度: {best.sam_alignment_score:.1f}")
            print(f"   邏輯一致性: {best.logical_consistency:.2f}")
            
            if hasattr(best, 'evolution_feedback') and best.evolution_feedback:
                print(f"   進化反饋: {best.evolution_feedback}")
            
            # 每3代顯示一次最佳論證
            if gen % 3 == 0:
                print(f"\n   🏆 當前最佳論證:")
                for i, p in enumerate(best.premises[:3], 1):
                    print(f"     前提{i}: {p.text[:50]}...")
                print(f"     結論: {best.claim.text[:60]}...")
    
    def _final_optimization(self):
        """最終優化階段"""
        print("\n" + "="*80)
        print("🎯 最終自洽優化階段")
        print("="*80)
        
        for ind in self.pop.population[:20]:  # 只優化前20個
            # 應用Sam導師的完整進化邏輯
            ind.mutate_with_sam_logic()
            
            # 重新評分
            EnhancedLLMFitnessEvaluator.evaluate_with_sam_philosophy(ind)
        
        # 排序
        self.pop.population.sort(key=lambda x: x.total_fitness, reverse=True)
    
    def _get_best_solution(self) -> Optional[EvolutionaryInference]:
        """獲取最佳解決方案"""
        if not self.pop.population:
            return None
        
        best = self.pop.population[0]
        
        # 驗證是否符合Sam導師思想
        is_valid, feedback = SamMetaValidator.validate_evolution_logic(best)
        
        print("\n" + "="*80)
        print("📋 Sam導師終極驗證報告")
        print("="*80)
        print(f"進化思想符合度: {'✅' if is_valid else '❌'} {feedback}")
        print(f"核心價值對齊度: {best.sam_alignment_score:.1f}")
        print(f"邏輯一致性: {best.logical_consistency:.2f}")
        print(f"總適應度: {best.total_fitness:.3f}")
        print(f"新穎性分數: {best.novelty_score:.3f}")
        print(f"世代: {best.generation}")
        
        print(f"\n🏆 最終最佳論證結構:")
        for i, p in enumerate(best.premises, 1):
            print(f"  前提{i}: {p.text}")
        print(f"  結論: {best.claim.text}")
        
        if hasattr(best.claim, 'superposition') and best.claim.superposition:
            print(f"  量子疊加態: {best.claim.superposition[0]}")
        
        return best

# =========================================================
# 主程序
# =========================================================

if __name__ == "__main__":
    print("🚀 啟動Sam導師思想增強版進化系統")
    print("="*80)
    
    # 初始化種群
    pop = PersistentPopulation("sam_enhanced_population.json")
    
    # 如果種群為空，注入初始種子
    if not pop.population:
        print("📦 注入初始種子論證...")
        initial_arguments = [
            "進化始於模仿\n模仿需要觀察\n觀察導致變異\n變異產生創造\n因此進化是創造之源",
            "實驗驗證真理\n真理需要檢驗\n檢驗依賴實驗\n因此實驗是真理的基石",
            "愛是核心價值\n和平體現愛\n創造需要和平環境\n因此愛促進創造",
            "邏輯需要自洽\n自洽需要驗證\n驗證需要實驗\n因此邏輯實驗密不可分",
            "量子疊加是可能態\n可能態需要觀察坍縮\n觀察創造現實\n因此觀察是創造行為"
        ]
        
        for arg in initial_arguments:
            inference = DiscourseParser.parse(arg)
            enhanced_premises = [EnhancedPremise(p.text) for p in inference.premises]
            enhanced_claim = QuantumClaim(inference.claim.text)
            enhanced_inference = EvolutionaryInference(enhanced_premises, enhanced_claim)
            pop.add(enhanced_inference)
    
    # 創建增強型進化引擎
    engine = EnhancedEvolutionEngine(pop)
    
    # 運行進化（帶自我批判和自洽優化）
    best = engine.evolve_with_self_criticism(generations=8, target_size=40)
    
    # 保存最終結果
    if best:
        result_file = "sam_final_evolution_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(best.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n💾 最終結果已保存至: {result_file}")
        
        # 顯示自我批判日誌
        if engine.self_criticism_log:
            print(f"\n📝 自我批判日誌（共{len(engine.self_criticism_log)}條）:")
            for i, log in enumerate(engine.self_criticism_log[-5:], 1):  # 顯示最後5條
                print(f"  {i}. {log['issue']}")
    
    print("\n" + "="*80)
    print("🎉 Sam導師思想增強版進化系統完成")
    print("="*80)