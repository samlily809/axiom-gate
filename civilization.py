"""
文明演化模块 v2.1 - 量子演化的數位文明（完全對齊修復版）
修复内容：
1. 能量恢復邏輯優化 - 防止過度恢復和量子狀態異常
2. 交互安全檢查增強 - 多層次檢查確保安全交互
3. 交互嘗試完整記錄 - 記錄所有嘗試便於調試
4. 對稱性拮抗精確檢測 - 改進量子坍縮決策邏輯
"""

import json
import uuid
import random
from datetime import datetime
from typing import Dict, List, Any, Optional


class CivilizationEntity:
    """
    文明實體 v2.1 - 增強穩定性修復
    新增：量子狀態平衡機制、勇氣自動回歸、交互安全檢查
    """
    
    def __init__(
        self,
        entity_id: str,
        name: str,
        responsibility: str,
        fear_of_loss: str,
        hard_limits: List[str],
        vision_boundary: str,
        can_be_negated: bool = True,
        quantum_state: float = 0.5  # 量子疊加狀態 (0=確定, 1=不確定)
    ):
        self.entity_id = entity_id
        self.name = name
        self.responsibility = responsibility
        self.fear_of_loss = fear_of_loss
        self.hard_limits = hard_limits
        self.vision_boundary = vision_boundary
        self.can_be_negated = can_be_negated
        self.quantum_state = max(0.1, min(0.9, quantum_state))  # 確保在合理範圍
        
        # 狀態屬性
        self.energy_level = 100.0
        self.autonomy_score = 0.0
        self.entropy_history: List[float] = []
        self.interaction_partners: List[str] = []
        self.evolution_contributions: int = 0
        
        # 量子演化屬性
        self.quantum_entangled_partners: List[str] = []  # 量子糾纏關係
        self.superposition_weights: Dict[str, float] = {}  # 狀態疊加權重
        self.collapse_history: List[Dict[str, Any]] = []  # 量子坍縮歷史
        self.courage_level: float = 0.5  # 勇氣水平 (0=保守, 1=勇敢)
        
        # 文明規則（從交互中學習）
        self.learned_rules: List[Dict[str, Any]] = []
        
        # 新增：狀態穩定性追蹤
        self.stability_score: float = 0.7  # 穩定性評分 (0=不穩定, 1=穩定)
        self.interaction_attempts: int = 0
        self.last_interaction_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "responsibility": self.responsibility,
            "fear_of_loss": self.fear_of_loss,
            "hard_limits": self.hard_limits,
            "vision_boundary": self.vision_boundary,
            "can_be_negated": self.can_be_negated,
            "quantum_state": round(self.quantum_state, 3),
            "energy_level": round(self.energy_level, 1),
            "autonomy_score": round(self.autonomy_score, 3),
            "current_entropy": round(self.get_current_entropy(), 3),
            "quantum_entropy": round(self.get_quantum_entropy(), 3),
            "courage_level": round(self.courage_level, 3),
            "stability_score": round(self.stability_score, 3),
            "interaction_count": len(self.interaction_partners),
            "evolution_contributions": self.evolution_contributions,
            "learned_rules_count": len(self.learned_rules),
            "entangled_partners_count": len(self.quantum_entangled_partners),
            "interaction_attempts": self.interaction_attempts
        }
    
    def get_current_entropy(self) -> float:
        """獲取當前熵值（最近5次交互的平均）"""
        if not self.entropy_history:
            return 0.5
        
        recent = self.entropy_history[-5:]
        return sum(recent) / len(recent)
    
    def get_quantum_entropy(self) -> float:
        """獲取量子熵值 - 衡量狀態不確定性"""
        # 量子熵 = 量子狀態 * (1 - 勇氣調整因子)
        courage_factor = 1.0 - abs(self.courage_level - 0.5) * 2  # 勇氣越極端，量子熵越低
        quantum_entropy = self.quantum_state * courage_factor
        return max(0.1, min(0.9, quantum_entropy))  # 確保在合理範圍
    
    def can_interact(self) -> bool:
        """檢查是否可以交互"""
        return (self.energy_level > 20.0 and 
                self.quantum_state < 0.9 and 
                self.stability_score > 0.3)
    
    def get_interaction_readiness(self) -> Dict[str, Any]:
        """獲取交互準備度詳情"""
        return {
            "energy_ok": self.energy_level > 20.0,
            "quantum_stable": self.quantum_state < 0.9,
            "stability_ok": self.stability_score > 0.3,
            "energy_level": self.energy_level,
            "quantum_state": self.quantum_state,
            "stability_score": self.stability_score
        }
    
    def consume_energy(self, amount: float):
        """消耗能量"""
        self.energy_level = max(0.0, self.energy_level - amount)
        
        # 能量消耗影響量子狀態和穩定性
        if self.energy_level < 30.0:
            self.quantum_state = min(0.9, self.quantum_state + 0.1)  # 低能量時更不確定
            self.stability_score = max(0.1, self.stability_score - 0.1)
    
    def gain_energy(self, amount: float):
        """獲得能量 - 修復版：防止過度恢復"""
        # 只在能量未滿時恢復
        if self.energy_level < 99.9:
            self.energy_level = min(100.0, self.energy_level + amount)
            
            # 能量恢復影響量子狀態和穩定性
            if self.energy_level > 70.0:
                self.quantum_state = max(0.1, self.quantum_state - 0.05)  # 高能量時更確定
                self.stability_score = min(1.0, self.stability_score + 0.05)
    
    def adjust_courage(self, adjustment: float):
        """調整勇氣水平 - 修復版：增加穩定性檢查"""
        old_courage = self.courage_level
        self.courage_level = max(0.0, min(1.0, self.courage_level + adjustment))
        
        # 記錄勇氣調整
        adjustment_magnitude = abs(adjustment)
        
        # 勇氣調整影響量子狀態
        if adjustment_magnitude > 0.1:  # 只有顯著調整才影響量子狀態
            if self.courage_level > 0.8:  # 過度勇敢
                self.quantum_state = min(0.9, self.quantum_state + 0.15)  # 變得更不確定
                self.stability_score = max(0.2, self.stability_score - 0.1)
                print(f"   ⚠️ {self.name} 勇氣過剩({self.courage_level:.2f})，量子不確定性增加")
            elif self.courage_level < 0.2:  # 過度保守
                self.quantum_state = max(0.1, self.quantum_state - 0.15)  # 變得更確定
                self.stability_score = max(0.2, self.stability_score - 0.1)
        
        # 勇氣自動回歸機制 (防止長期極端化)
        if old_courage > 0.8 and self.courage_level > 0.8:
            # 持續高勇氣時輕微降低
            self.courage_level = max(0.7, self.courage_level - 0.01)
        elif old_courage < 0.2 and self.courage_level < 0.2:
            # 持續低勇氣時輕微提高
            self.courage_level = min(0.3, self.courage_level + 0.01)
    
    def record_interaction(self, partner_id: str, entropy: float, courage_adjustment: float = 0.0):
        """記錄交互"""
        if partner_id not in self.interaction_partners:
            self.interaction_partners.append(partner_id)
        
        self.entropy_history.append(entropy)
        self.last_interaction_time = datetime.now().isoformat()
        
        # 調整勇氣
        if courage_adjustment != 0.0:
            self.adjust_courage(courage_adjustment)
        
        # 更新穩定性分數 (成功交互提升穩定性)
        if entropy > 0.3 and entropy < 0.7:  # 中等熵值交互最穩定
            self.stability_score = min(1.0, self.stability_score + 0.05)
        
        # 保持歷史長度
        if len(self.entropy_history) > 100:
            self.entropy_history = self.entropy_history[-100:]
        
        # 記錄量子坍縮
        collapse_event = {
            "timestamp": self.last_interaction_time,
            "partner": partner_id,
            "entropy": entropy,
            "quantum_state_before": self.quantum_state,
            "courage_level": self.courage_level,
            "stability_score": self.stability_score
        }
        self.collapse_history.append(collapse_event)
        
        if len(self.collapse_history) > 50:
            self.collapse_history = self.collapse_history[-50:]
    
    def record_interaction_attempt(self, success: bool, reason: str = ""):
        """記錄交互嘗試"""
        self.interaction_attempts += 1
        
        if not success:
            # 失敗嘗試降低穩定性
            self.stability_score = max(0.1, self.stability_score - 0.02)
        else:
            # 成功嘗試略微提升穩定性
            self.stability_score = min(1.0, self.stability_score + 0.01)


class CivilizationEngine:
    """
    文明演化引擎 v2.1 - 完全對齊修復版
    修復重點：交互安全檢查、能量恢復邏輯、對稱性拮抗處理
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.entities: Dict[str, CivilizationEntity] = {}
        self.civilization_rules: List[Dict[str, Any]] = []
        self.interaction_history: List[Dict[str, Any]] = []
        self.evolution_cycles: int = 0
        self.decision_cache: List[Dict[str, Any]] = []
        
        # 對稱性拮抗修復參數
        self.defer_count = 0
        self.entity_weights = {
            'ENTITY_ACTION': 1.0,
            'ENTITY_NEGATION': 1.0,
            'ENTITY_VALUE': 1.2  # 初始價值權重提升
        }
        self.symmetry_threshold = 0.05  # 對稱性檢測閾值
        self.courage_correction_active = False
        
        # 量子演化參數
        self.quantum_entanglement_network: Dict[str, List[str]] = {}
        self.superposition_field: float = 0.5  # 全局疊加場
        
        # 新增：系統穩定性追蹤
        self.system_stability: float = 0.8
        self.interaction_attempt_log: List[Dict[str, Any]] = []
        self.blocked_interactions: int = 0
        
        # 初始化基礎規則
        self._initialize_base_rules()
    
    def _initialize_base_rules(self):
        """初始化基礎文明規則"""
        base_rules = [
            {
                "rule_id": "RULE_BASE_001",
                "type": "INTERACTION",
                "content": "所有交互必須可追溯",
                "source": "SYSTEM_INIT",
                "strength": 1.0,
                "conflict_keywords": ["不可追溯", "隱藏交互"]
            },
            {
                "rule_id": "RULE_BASE_002",
                "type": "ENERGY",
                "content": "能量低於20的實體不得發起新交互",
                "source": "SYSTEM_INIT",
                "strength": 1.0,
                "conflict_keywords": ["無視能量", "強制交互"]
            },
            {
                "rule_id": "RULE_BASE_003",
                "type": "EVOLUTION",
                "content": "新規則必須與現有規則無根本衝突",
                "source": "SYSTEM_INIT",
                "strength": 0.9,
                "conflict_keywords": ["違反現有規則", "邏輯矛盾"]
            },
            {
                "rule_id": "RULE_BASE_004",
                "type": "QUANTUM",
                "content": "量子狀態高於0.8的實體應優先進行糾纏交互",
                "source": "SYSTEM_INIT",
                "strength": 0.7,
                "conflict_keywords": ["忽略量子狀態", "隨機交互"]
            },
            {
                "rule_id": "RULE_BASE_005",
                "type": "SYMMETRY",
                "content": "檢測到對稱性拮抗時，應啟動價值優先裁決",
                "source": "SYSTEM_INIT",
                "strength": 0.8,
                "conflict_keywords": ["忽略對稱性", "隨機裁決"]
            },
            {
                "rule_id": "RULE_BASE_006",
                "type": "SAFETY",
                "content": "交互前必須通過多重安全檢查",
                "source": "SYSTEM_INIT",
                "strength": 0.9,
                "conflict_keywords": ["跳過檢查", "強制執行"]
            }
        ]
        
        self.civilization_rules.extend(base_rules)
    
    def create_entity(self, entity_config: Dict[str, Any]) -> CivilizationEntity:
        """創建新的文明實體"""
        entity_id = entity_config.get("entity_id", str(uuid.uuid4())[:8])
        
        entity = CivilizationEntity(
            entity_id=entity_id,
            name=entity_config["name"],
            responsibility=entity_config["responsibility"],
            fear_of_loss=entity_config["fear_of_loss"],
            hard_limits=entity_config["hard_limits"],
            vision_boundary=entity_config["vision_boundary"],
            can_be_negated=entity_config.get("can_be_negated", True),
            quantum_state=entity_config.get("quantum_state", 0.5)
        )
        
        # 根據責任類型初始化勇氣水平
        if "邏輯" in entity_config["responsibility"]:
            entity.courage_level = 0.7  # 邏輯實體較勇敢
        elif "穩定" in entity_config["responsibility"]:
            entity.courage_level = 0.3  # 穩定實體較保守
        elif "價值" in entity_config["responsibility"]:
            entity.courage_level = 0.5  # 價值實體中性
        
        self.entities[entity_id] = entity
        print(f"🏛️ 創建量子文明實體：{entity.name} ({entity_id})")
        print(f"   量子狀態：{entity.quantum_state:.2f}, 勇氣水平：{entity.courage_level:.2f}")
        print(f"   穩定性：{entity.stability_score:.2f}, 能量：{entity.energy_level:.1f}")
        
        return entity
    
    def inject_rule(self, axiom_id: str, content: str, rule_type: str = "INJECTED"):
        """對齊 AXIOM_005：允許外部指令注入新規則以打破僵局"""
        # 創建臨時規則來檢查衝突
        temp_rule = {
            "content": content,
            "conflict_keywords": self._extract_conflict_keywords(content)
        }
        
        # 檢查規則是否與現有規則衝突
        if self._is_rule_conflicting(temp_rule):
            print(f"⚠️ 規則注入衝突：{content}")
            return None
        
        new_rule = {
            "rule_id": f"RULE_INJECT_{axiom_id}_{uuid.uuid4().hex[:4]}",
            "type": rule_type,
            "content": content,
            "source": "COMMANDER_INTERVENTION",
            "timestamp": datetime.now().isoformat(),
            "strength": 0.9,
            "acceptance_score": 0.0,
            "conflict_keywords": self._extract_conflict_keywords(content)
        }
        
        self.civilization_rules.append(new_rule)
        print(f"📜 [外部注入] 成功：{content}")
        
        # 通知所有實體學習新規則
        for entity in self.entities.values():
            entity.learned_rules.append(new_rule)
        
        # 更新系統穩定性
        self.system_stability = min(1.0, self.system_stability + 0.02)
        
        return new_rule
    
    def process_miniasi_decision(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        處理MiniASI的決策輸出 v2.1 - 增強穩定性修復
        """
        status = decision_data.get('status', 'UNKNOWN')
        entropy = decision_data.get('entropy', 0.5)
        input_text = decision_data.get('input', '未指定輸入')
        symmetry_detected = decision_data.get('symmetry_detected', False)
        courage_adjustment = decision_data.get('courage_adjustment', 1.0)
        
        print(f"🔄 處理MiniASI決策: {input_text[:50]}...")
        print(f"   狀態: {status}, 熵值: {entropy:.3f}")
        print(f"   對稱性檢測: {symmetry_detected}, 勇氣調整: {courage_adjustment}")
        
        # 檢測僵局並更新計數
        if status == "DEFERRED" and symmetry_detected:
            self.defer_count += 1
            print(f"   🔁 連續僵局計數: {self.defer_count}")
            
            # 觸發AXIOM_006：停滯即死亡
            if self.defer_count >= 2:
                print("⚡ [AXIOM_006觸發] 偵測到演化停滯，注入動態擾動...")
                
                # 動態調整價值權重
                self.entity_weights['ENTITY_VALUE'] *= 1.1
                print(f"   價值權重調整為: {self.entity_weights['ENTITY_VALUE']:.2f}")
                
                # 激活勇氣修正
                self.courage_correction_active = True
                
                # 降低系統穩定性（停滯處罰）
                self.system_stability = max(0.3, self.system_stability - 0.1)
                
                # 注入外部規則打破僵局
                new_rule = self.inject_rule(
                    "006_BREAKER",
                    "連續兩次決策僵局時，必須引入外部隨機性或調整實體權重以打破平衡",
                    "DYNAMIC_PERTURBATION"
                )
                
                if new_rule:
                    # 重置僵局計數
                    self.defer_count = 0
                    # 恢復系統穩定性
                    self.system_stability = min(1.0, self.system_stability + 0.05)
        else:
            self.defer_count = 0
            self.courage_correction_active = False
            # 正常決策提升穩定性
            self.system_stability = min(1.0, self.system_stability + 0.02)
        
        # 緩存決策供後續使用
        cache_entry = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision_data,
            "processed": False,
            "symmetry": symmetry_detected,
            "quantum_entropy": entropy,
            "system_stability": self.system_stability
        }
        
        self.decision_cache.append(cache_entry)
        
        # 限制緩存大小
        if len(self.decision_cache) > 20:
            self.decision_cache = self.decision_cache[-20:]
        
        # 根據決策狀態採取行動
        if status in ["EXECUTED", "ACCEPTED"]:
            return self._handle_accepted_decision(decision_data)
        elif status in ["DEFERRED", "REJECTED"]:
            return self._handle_deferred_decision(decision_data)
        else:
            return {
                "status": "UNPROCESSED",
                "reason": f"未知狀態: {status}",
                "action": "跳過處理",
                "system_stability": self.system_stability
            }
    
    def _handle_accepted_decision(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """處理被接受的決策"""
        if not self.entities:
            return {"status": "NO_ENTITIES", "action": "跳過"}
        
        # 選擇合適的實體進行交互（考慮量子狀態和穩定性）
        available_entities = [e for e in self.entities.values() if e.can_interact()]
        if not available_entities:
            print("   ⚠️ 無可用實體：檢查能量和穩定性")
            for entity in self.entities.values():
                readiness = entity.get_interaction_readiness()
                print(f"      {entity.name}: {readiness}")
            return {"status": "NO_ENERGY", "action": "等待能量恢復"}
        
        decision_entropy = decision_data.get('entropy', 0.5)
        symmetry_detected = decision_data.get('symmetry_detected', False)
        
        # 選擇實體的優先級邏輯
        if symmetry_detected and 'ENTITY_VALUE' in [e.entity_id for e in available_entities]:
            # 對稱性情況下優先選擇價值實體
            value_entities = [e for e in available_entities if 'VALUE' in e.entity_id]
            if value_entities:
                entity1 = max(value_entities, key=lambda e: e.stability_score)
                print(f"   🎯 對稱性檢測，選擇最穩定價值實體: {entity1.name}")
        else:
            # 正常情況下選擇量子熵值和穩定性平衡最佳的實體
            entity1 = max(
                available_entities,
                key=lambda e: (e.get_quantum_entropy() * 0.6 + 
                              e.stability_score * 0.4) * 
                              self.entity_weights.get(e.entity_id, 1.0)
            )
        
        # 尋找交互夥伴
        entity2 = self.find_interaction_partner(entity1, decision_entropy)
        if not entity2:
            # 記錄找不到夥伴的嘗試
            attempt_record = self._record_interaction_attempt(
                entity1, None, "尋找交互夥伴", 
                "NO_PARTNER_FOUND", "未找到合適的交互夥伴"
            )
            return {
                "status": "NO_PARTNER", 
                "action": "等待夥伴可用",
                "attempt_record": attempt_record
            }
        
        # 執行交互前檢查
        safety_check = self._safety_check_interaction(entity1, entity2)
        if not safety_check["safe"]:
            # 記錄安全檢查失敗
            attempt_record = self._record_interaction_attempt(
                entity1, entity2, decision_data.get('input', '未指定話題'),
                "SAFETY_CHECK_FAILED", safety_check["reasons"]
            )
            return {
                "status": "SAFETY_BLOCKED",
                "reasons": safety_check["reasons"],
                "action": "跳過不安全交互",
                "attempt_record": attempt_record
            }
        
        # 計算勇氣調整（如果激活了勇氣修正）
        courage_adjustment = 0.0
        if self.courage_correction_active:
            # 如果實體過度勇敢，施加負向調整
            if entity1.courage_level > 0.7:
                courage_adjustment = -0.1
                print(f"   🛡️ 對{entity1.name}應用勇氣降溫: -0.1")
            elif entity2.courage_level > 0.7:
                courage_adjustment = -0.1
                print(f"   🛡️ 對{entity2.name}應用勇氣降溫: -0.1")
        
        # 執行交互
        topic = decision_data.get('input', '未指定話題')
        interaction_result = self.entity_interaction(
            entity1, entity2, topic, 
            courage_adjustment=courage_adjustment,
            symmetry_context=symmetry_detected
        )
        
        return {
            "status": "PROCESSED",
            "interaction_result": interaction_result,
            "entities_involved": [entity1.name, entity2.name],
            "courage_adjustment_applied": courage_adjustment,
            "safety_check_passed": True,
            "system_stability": self.system_stability
        }
    
    def _handle_deferred_decision(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """處理被推遲的決策 - 生成新規則（修復版）"""
        results = decision_data.get('results', {})
        if not results:
            return {
                "status": "NO_RESULTS", 
                "action": "跳過",
                "system_stability": self.system_stability
            }
        
        # 分析衝突
        collision_points = []
        symmetry_level = 0.5
        
        if "ACTION" in results and "NEGATION" in results:
            action_text = results.get("ACTION", "")
            negation_text = results.get("NEGATION", "")
            
            # 提取分數（如果有）
            scores = decision_data.get('scores', {})
            action_score = scores.get("ACTION", 0.5)
            negation_score = scores.get("NEGATION", 0.5)
            
            diff = abs(action_score - negation_score)
            symmetry_level = diff
            
            if diff < self.symmetry_threshold:
                collision_points.append(f"對稱性拮抗 (Δ={diff:.3f})")
            
            # 內容衝突檢測
            conflict_pairs = [
                ("接受", "拒絕"),
                ("執行", "停止"),
                ("肯定", "否定"),
                ("前進", "後退"),
                ("允許", "禁止"),
                ("是", "否")
            ]
            
            for positive, negative in conflict_pairs:
                if positive in action_text and negative in negation_text:
                    collision_points.append(f"{positive} vs {negative}")
        
        if collision_points:
            # 生成新規則來解決衝突
            new_rule = self._generate_rule_from_conflict(
                collision_points,
                decision_data.get('input', '未指定輸入'),
                symmetry_level=symmetry_level
            )
            
            if new_rule:
                return {
                    "status": "RULE_GENERATED",
                    "rule": new_rule,
                    "collision_points": collision_points,
                    "symmetry_detected": symmetry_level < self.symmetry_threshold,
                    "symmetry_level": symmetry_level,
                    "system_stability": self.system_stability
                }
        
        return {
            "status": "DEFERRED_NO_CONFLICT", 
            "action": "等待更多輸入",
            "system_stability": self.system_stability
        }
    
    def find_interaction_partner(self, entity: CivilizationEntity, target_entropy: float = 0.5) -> Optional[CivilizationEntity]:
        """為實體尋找交互夥伴 v2.1 - 增強匹配算法"""
        possible_partners = []
        
        for other_id, other_entity in self.entities.items():
            if other_id == entity.entity_id:
                continue
            
            # 跳過不能交互的實體
            if not other_entity.can_interact():
                continue
            
            # 檢查是否已多次交互（避免重複交互）
            if other_id in entity.interaction_partners[-3:]:
                continue
            
            # 檢查硬限制衝突
            hard_limit_conflict = False
            for limit1 in entity.hard_limits:
                for limit2 in other_entity.hard_limits:
                    if self._are_limits_conflicting(limit1, limit2):
                        hard_limit_conflict = True
                        break
                if hard_limit_conflict:
                    break
            
            if hard_limit_conflict:
                continue
            
            # 計算匹配度分數
            priority = 1.0
            
            # 量子糾纏夥伴優先
            if other_id in entity.quantum_entangled_partners:
                priority = 2.0
            
            # 計算各項兼容性
            entropy_diff = abs(entity.get_current_entropy() - other_entity.get_current_entropy())
            quantum_compatibility = 1.0 - abs(entity.quantum_state - other_entity.quantum_state)
            courage_balance = 1.0 - abs(entity.courage_level - 0.5) * abs(other_entity.courage_level - 0.5)
            stability_compatibility = (entity.stability_score + other_entity.stability_score) / 2
            
            # 綜合匹配分數
            match_score = (
                (1.0 - entropy_diff) * 0.3 +          # 熵值兼容性
                quantum_compatibility * 0.25 +        # 量子兼容性
                courage_balance * 0.2 +               # 勇氣平衡
                stability_compatibility * 0.25        # 穩定性兼容性
            ) * priority
            
            possible_partners.append((other_entity, match_score))
        
        if not possible_partners:
            return None
        
        # 選擇匹配度最高的夥伴
        possible_partners.sort(key=lambda x: x[1], reverse=True)
        best_partner = possible_partners[0][0]
        
        # 建立量子糾纏（如果匹配度很高且雙方都穩定）
        if (possible_partners[0][1] > 1.5 and 
            entity.stability_score > 0.6 and 
            best_partner.stability_score > 0.6 and
            entity.entity_id not in best_partner.quantum_entangled_partners):
            
            entity.quantum_entangled_partners.append(best_partner.entity_id)
            best_partner.quantum_entangled_partners.append(entity.entity_id)
            
            # 更新量子糾纏網絡
            if entity.entity_id not in self.quantum_entanglement_network:
                self.quantum_entanglement_network[entity.entity_id] = []
            if best_partner.entity_id not in self.quantum_entanglement_network:
                self.quantum_entanglement_network[best_partner.entity_id] = []
            
            self.quantum_entanglement_network[entity.entity_id].append(best_partner.entity_id)
            self.quantum_entanglement_network[best_partner.entity_id].append(entity.entity_id)
            
            print(f"   🔗 {entity.name} 與 {best_partner.name} 建立量子糾纏")
            print(f"      匹配分數: {possible_partners[0][1]:.2f}, 穩定性: {entity.stability_score:.2f}/{best_partner.stability_score:.2f}")
        
        return best_partner
    
    def _safety_check_interaction(self, entity1: CivilizationEntity, entity2: CivilizationEntity) -> Dict[str, Any]:
        """交互安全檢查 v2.1 - 多重檢查確保安全"""
        checks = []
        safe = True
        
        # 檢查1: 能量檢查
        if entity1.energy_level < 25.0 or entity2.energy_level < 25.0:
            checks.append(f"能量不足: {entity1.name}({entity1.energy_level:.1f}), {entity2.name}({entity2.energy_level:.1f})")
            safe = False
        
        # 檢查2: 量子狀態檢查
        if entity1.quantum_state > 0.85 or entity2.quantum_state > 0.85:
            checks.append(f"量子狀態過高: {entity1.name}({entity1.quantum_state:.2f}), {entity2.name}({entity2.quantum_state:.2f})")
            safe = False
        
        # 檢查3: 穩定性檢查
        if entity1.stability_score < 0.4 or entity2.stability_score < 0.4:
            checks.append(f"穩定性不足: {entity1.name}({entity1.stability_score:.2f}), {entity2.name}({entity2.stability_score:.2f})")
            safe = False
        
        # 檢查4: 勇氣差異檢查
        courage_diff = abs(entity1.courage_level - entity2.courage_level)
        if courage_diff > 0.7:
            checks.append(f"勇氣差異過大: Δ={courage_diff:.2f}")
            safe = False
        
        # 檢查5: 硬限制衝突
        hard_limit_conflict = False
        for limit1 in entity1.hard_limits:
            for limit2 in entity2.hard_limits:
                if self._are_limits_conflicting(limit1, limit2):
                    checks.append(f"硬限制衝突: {limit1} vs {limit2}")
                    hard_limit_conflict = True
                    safe = False
                    break
            if hard_limit_conflict:
                break
        
        # 檢查6: 規則衝突
        rule_violations = self._check_interaction_rules(entity1, entity2)
        if rule_violations:
            checks.extend(rule_violations)
            safe = False
        
        return {
            "safe": safe,
            "reasons": checks,
            "details": {
                "energy": [entity1.energy_level, entity2.energy_level],
                "quantum": [entity1.quantum_state, entity2.quantum_state],
                "stability": [entity1.stability_score, entity2.stability_score],
                "courage_diff": courage_diff
            }
        }
    
    def entity_interaction(
        self,
        entity1: CivilizationEntity,
        entity2: CivilizationEntity,
        topic: str = "未指定話題",
        courage_adjustment: float = 0.0,
        symmetry_context: bool = False
    ) -> Dict[str, Any]:
        """執行實體間的交互 v2.1 - 完全安全修復版"""
        
        # 記錄交互嘗試開始
        entity1.record_interaction_attempt(True, "開始交互")
        entity2.record_interaction_attempt(True, "開始交互")
        
        # 安全檢查（雙重檢查確保安全）
        safety_check = self._safety_check_interaction(entity1, entity2)
        if not safety_check["safe"]:
            # 記錄失敗嘗試
            entity1.record_interaction_attempt(False, "安全檢查失敗")
            entity2.record_interaction_attempt(False, "安全檢查失敗")
            
            # 記錄到交互嘗試日誌
            self._record_interaction_attempt(
                entity1, entity2, topic,
                "SAFETY_CHECK_FAILED", safety_check["reasons"]
            )
            
            return {
                "status": "BLOCKED",
                "reason": "交互安全檢查失敗",
                "safety_checks": safety_check["reasons"],
                "details": safety_check["details"],
                "attempt_recorded": True
            }
        
        # 計算交互熵值
        entropy1 = entity1.get_current_entropy()
        entropy2 = entity2.get_current_entropy()
        quantum_entropy1 = entity1.get_quantum_entropy()
        quantum_entropy2 = entity2.get_quantum_entropy()
        
        interaction_entropy = (entropy1 + entropy2) / 2
        quantum_interaction_entropy = (quantum_entropy1 + quantum_entropy2) / 2
        
        # 執行交互邏輯
        print(f"🤝 {entity1.name} 與 {entity2.name} 進行量子交互...")
        print(f"   話題：{topic}")
        print(f"   交互熵：{interaction_entropy:.2f}, 量子交互熵：{quantum_interaction_entropy:.2f}")
        print(f"   勇氣水平：{entity1.courage_level:.2f} vs {entity2.courage_level:.2f}")
        print(f"   穩定性：{entity1.stability_score:.2f} vs {entity2.stability_score:.2f}")
        
        # 外交協調嘗試
        coordination = self._diplomatic_coordination(entity1, entity2, topic)
        
        # 辨證碰撞檢測
        collision_points = self._detect_collision_points(entity1, entity2, topic)
        
        # 對稱性拮抗特殊處理
        if symmetry_context:
            collision_points.append("對稱性拮抗上下文")
            print(f"   ⚖️ 對稱性上下文：啟用特殊處理")
            
            # 在對稱性上下文中，鼓勵勇氣平衡
            if entity1.courage_level > 0.7 and entity2.courage_level < 0.3:
                courage_adjustment = -0.15  # 降低勇敢方的勇氣
                print(f"   🛡️ 對{entity1.name}應用對稱性勇氣降溫: -0.15")
            elif entity2.courage_level > 0.7 and entity1.courage_level < 0.3:
                courage_adjustment = -0.15
                print(f"   🛡️ 對{entity2.name}應用對稱性勇氣降溫: -0.15")
        
        # 構建交互結果
        interaction_result = {
            "interaction_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "participants": [entity1.entity_id, entity2.entity_id],
            "participant_names": [entity1.name, entity2.name],
            "topic": topic,
            "interaction_entropy": round(interaction_entropy, 3),
            "quantum_interaction_entropy": round(quantum_interaction_entropy, 3),
            "coordination_result": coordination,
            "collision_points": collision_points,
            "courage_levels": [round(entity1.courage_level, 3), round(entity2.courage_level, 3)],
            "stability_scores": [round(entity1.stability_score, 3), round(entity2.stability_score, 3)],
            "new_rule_generated": False,
            "status": "SUCCESS",
            "energy_cost": 15.0,
            "symmetry_context": symmetry_context,
            "quantum_entangled": entity2.entity_id in entity1.quantum_entangled_partners,
            "applied_courage_adjustment": courage_adjustment,
            "safety_check_passed": True,
            "system_stability_before": self.system_stability
        }
        
        # 量子坍縮：根據交互結果更新量子狀態
        if coordination["success"]:
            # 協調成功，量子狀態趨向確定，穩定性提高
            entity1.quantum_state = max(0.1, entity1.quantum_state - 0.1)
            entity2.quantum_state = max(0.1, entity2.quantum_state - 0.1)
            entity1.stability_score = min(1.0, entity1.stability_score + 0.05)
            entity2.stability_score = min(1.0, entity2.stability_score + 0.05)
            self.system_stability = min(1.0, self.system_stability + 0.03)
        else:
            # 協調失敗，量子狀態趨向不確定，穩定性降低
            entity1.quantum_state = min(0.9, entity1.quantum_state + 0.1)
            entity2.quantum_state = min(0.9, entity2.quantum_state + 0.1)
            entity1.stability_score = max(0.1, entity1.stability_score - 0.03)
            entity2.stability_score = max(0.1, entity2.stability_score - 0.03)
            self.system_stability = max(0.3, self.system_stability - 0.02)
        
        # 檢查是否產生新規則
        rule_generation_threshold = 0.6 if not symmetry_context else 0.4
        if collision_points and interaction_entropy > rule_generation_threshold:
            new_rule = self._generate_new_rule(entity1, entity2, collision_points, symmetry_context)
            if new_rule and not self._is_rule_conflicting(new_rule):
                interaction_result["new_rule_generated"] = True
                interaction_result["new_rule"] = new_rule
                
                # 記錄規則演化貢獻
                entity1.evolution_contributions += 1
                entity2.evolution_contributions += 1
                
                # 新規則提升系統穩定性
                self.system_stability = min(1.0, self.system_stability + 0.05)
                print(f"   📜 生成新規則：{new_rule['rule_id']}，系統穩定性+0.05")
        
        # 更新實體狀態
        energy_cost = interaction_result["energy_cost"]
        entity1.consume_energy(energy_cost)
        entity2.consume_energy(energy_cost)
        
        entity1.record_interaction(entity2.entity_id, interaction_entropy, courage_adjustment)
        entity2.record_interaction(entity1.entity_id, interaction_entropy, courage_adjustment)
        
        # 記錄交互歷史
        self.interaction_history.append(interaction_result)
        
        # 更新系統狀態
        interaction_result["system_stability_after"] = self.system_stability
        
        return interaction_result
    
    def _diplomatic_coordination(
        self,
        entity1: CivilizationEntity,
        entity2: CivilizationEntity,
        topic: str
    ) -> Dict[str, Any]:
        """外交協調 v2.1 - 考慮穩定性平衡"""
        # 計算協調基礎分數
        entropy_diff = abs(entity1.get_current_entropy() - entity2.get_current_entropy())
        quantum_diff = abs(entity1.quantum_state - entity2.quantum_state)
        courage_diff = abs(entity1.courage_level - entity2.courage_level)
        stability_avg = (entity1.stability_score + entity2.stability_score) / 2
        
        # 勇氣差異過大不利於協調
        courage_penalty = max(0, courage_diff - 0.3) * 2
        
        # 穩定性對協調的影響
        stability_bonus = (stability_avg - 0.5) * 0.3
        
        base_success_chance = 0.7 - entropy_diff - quantum_diff * 0.5 - courage_penalty + stability_bonus
        
        # 量子糾纏提升協調成功率
        if entity2.entity_id in entity1.quantum_entangled_partners:
            base_success_chance += 0.2
        
        # 確保成功率在合理範圍
        base_success_chance = max(0.1, min(0.9, base_success_chance))
        
        coordination_success = random.random() < base_success_chance
        
        coordination_methods = [
            "量子語義協商",
            "立場平衡對話",
            "價值共識建立",
            "風險分擔協議"
        ]
        
        return {
            "success": coordination_success,
            "coordination_level": random.uniform(0.4, 0.9) if coordination_success else random.uniform(0.1, 0.4),
            "method": random.choice(coordination_methods) if coordination_success else "立場堅持",
            "entropy_diff": round(entropy_diff, 3),
            "quantum_diff": round(quantum_diff, 3),
            "courage_diff": round(courage_diff, 3),
            "stability_avg": round(stability_avg, 3),
            "success_chance": round(base_success_chance, 3)
        }
    
    def _detect_collision_points(
        self,
        entity1: CivilizationEntity,
        entity2: CivilizationEntity,
        topic: str
    ) -> List[str]:
        """檢測碰撞點 v2.1 - 增強檢測邏輯"""
        collision_points = []
        
        # 基於責任差異
        if entity1.responsibility != entity2.responsibility:
            collision_points.append(f"責任差異：{entity1.responsibility} vs {entity2.responsibility}")
        
        # 基於量子狀態差異
        quantum_diff = abs(entity1.quantum_state - entity2.quantum_state)
        if quantum_diff > 0.3:
            collision_points.append(f"量子狀態差異：{quantum_diff:.2f}")
        
        # 基於勇氣水平差異
        courage_diff = abs(entity1.courage_level - entity2.courage_level)
        if courage_diff > 0.4:
            collision_points.append(f"勇氣水平差異：{courage_diff:.2f}")
            
            # 檢測勇氣過剩
            if entity1.courage_level > 0.7:
                collision_points.append(f"{entity1.name}勇氣過剩({entity1.courage_level:.2f})")
            if entity2.courage_level > 0.7:
                collision_points.append(f"{entity2.name}勇氣過剩({entity2.courage_level:.2f})")
        
        # 熵值差異
        entropy_diff = abs(entity1.get_current_entropy() - entity2.get_current_entropy())
        if entropy_diff > 0.3:
            collision_points.append(f"熵值差異：{entropy_diff:.2f}")
        
        # 穩定性差異
        stability_diff = abs(entity1.stability_score - entity2.stability_score)
        if stability_diff > 0.3:
            collision_points.append(f"穩定性差異：{stability_diff:.2f}")
        
        # 隨機添加一些碰撞點（模擬量子不確定性）
        quantum_collision_chance = entity1.quantum_state * entity2.quantum_state
        if random.random() < quantum_collision_chance:
            quantum_collisions = [
                "量子方法論分歧",
                "疊加態價值權重差異",
                "糾纏風險評估不一致",
                "坍縮時間偏好衝突",
                "不確定性容忍度差異",
                "量子路徑選擇分歧"
            ]
            collision_points.append(random.choice(quantum_collisions))
        
        return collision_points
    
    def _check_interaction_rules(
        self,
        entity1: CivilizationEntity,
        entity2: CivilizationEntity
    ) -> List[str]:
        """檢查交互是否違反文明規則"""
        violations = []
        
        # 檢查基礎規則
        for rule in self.civilization_rules:
            if rule["type"] == "INTERACTION":
                # 檢查交互規則
                if "必須可追溯" in rule["content"]:
                    # 確保交互可追溯（通過interaction_id）
                    pass  # 會在entity_interaction中處理
                    
            elif rule["type"] == "ENERGY":
                # 檢查能量規則
                if entity1.energy_level < 20.0 or entity2.energy_level < 20.0:
                    violations.append(f"違反能量規則：實體能量低於20")
            
            elif rule["type"] == "QUANTUM":
                # 檢查量子規則
                if entity1.quantum_state > 0.9 and entity2.quantum_state > 0.9:
                    if random.random() > 0.5:
                        violations.append("量子不確定性過高，交互可能產生不可預測結果")
        
        return violations
    
    def _are_limits_conflicting(self, limit1: str, limit2: str) -> bool:
        """檢查兩個限制是否衝突"""
        conflicts = [
            ("不能違反邏輯", "可以創造性跳躍"),
            ("必須保守", "必須激進"),
            ("禁止改變", "必須演化"),
            ("永遠確定", "擁抱不確定"),
            ("避免風險", "接受風險"),
            ("不能放棄", "可以妥協"),
            ("必須一致", "允許矛盾"),
            ("禁止否定", "必須質疑")
        ]
        
        for conflict_pair in conflicts:
            if (limit1 in conflict_pair and limit2 in conflict_pair):
                return True
        
        return False
    
    def _is_rule_conflicting(self, new_rule: Dict[str, Any]) -> bool:
        """檢查新規則是否與現有規則存在根本性衝突"""
        new_content = new_rule.get("content", "").lower()
        
        # 1. 檢查與現有規則的關鍵詞衝突
        for rule in self.civilization_rules:
            if "conflict_keywords" in rule:
                for keyword in rule["conflict_keywords"]:
                    if keyword in new_content:
                        print(f"   ⚠️ 規則衝突檢測：新規則包含衝突關鍵詞 '{keyword}'")
                        return True
        
        # 2. 邏輯矛盾檢測
        negation_words = ["不", "否", "無", "拒絕", "停止", "禁止", "避免", "禁止"]
        affirmation_words = ["是", "必須", "應該", "要求", "執行", "保持", "接受", "允許"]
        
        has_negation = any(word in new_content for word in negation_words)
        has_affirmation = any(word in new_content for word in affirmation_words)
        
        # 如果同時包含強烈的肯定和否定詞彙，可能邏輯矛盾
        strong_negations = ["不能", "禁止", "拒絕", "停止"]
        strong_affirmations = ["必須", "要求", "強制", "執行"]
        
        has_strong_negation = any(word in new_content for word in strong_negations)
        has_strong_affirmation = any(word in new_content for word in strong_affirmations)
        
        if has_strong_negation and has_strong_affirmation:
            print(f"   ⚠️ 規則衝突檢測：新規則同時包含強烈肯定和否定詞彙")
            return True
        
        # 3. 檢查是否直接否定基礎規則
        base_rules = [r for r in self.civilization_rules if r["source"] == "SYSTEM_INIT"]
        for rule in base_rules:
            rule_content = rule["content"].lower()
            # 簡單檢測：如果新規則包含"不" + 基礎規則關鍵詞
            for keyword in ["必須", "應該", "要求"]:
                if keyword in rule_content:
                    # 提取關鍵部分
                    key_part = rule_content.split(keyword)[-1].strip()[:20]
                    if key_part and f"不{key_part}" in new_content:
                        print(f"   ⚠️ 規則衝突檢測：新規則否定基礎規則 '{rule_content}'")
                        return True
        
        return False
    
    def _extract_conflict_keywords(self, content: str) -> List[str]:
        """從規則內容提取衝突關鍵詞"""
        content_lower = content.lower()
        conflict_keywords = []
        
        negation_words = ["不", "否", "無", "拒絕", "停止", "禁止", "避免"]
        for word in negation_words:
            if word in content_lower:
                conflict_keywords.append(word)
        
        # 添加特定的衝突模式
        if "不能" in content_lower and "必須" in content_lower:
            conflict_keywords.append("邏輯矛盾")
        
        return conflict_keywords
    
    def _generate_rule_from_conflict(self, collision_points: List[str], context: str, symmetry_level: float = 0.5) -> Dict[str, Any]:
        """從衝突中生成新規則 v2.1 - 考慮對稱性"""
        rule_id = f"RULE_SYMMETRY_{len(self.civilization_rules)+1:03d}"
        
        # 根據對稱性水平調整規則強度
        rule_strength = 0.6 + (1.0 - symmetry_level) * 0.3  # 對稱性越低，規則強度越高
        
        # 根據衝突點生成規則內容
        if "對稱性拮抗" in str(collision_points):
            rule_content = f"解決對稱性拮抗：當行動與否定分數差異小於{self.symmetry_threshold}時，啟動價值優先裁決"
        elif "勇氣過剩" in str(collision_points):
            rule_content = f"平衡勇氣水平：當實體勇氣超過0.7時，自動應用勇氣降溫調整"
        else:
            rule_content = f"解決衝突：{', '.join(collision_points[:2])} | 上下文：{context[:50]}..."
        
        new_rule = {
            "rule_id": rule_id,
            "type": "SYMMETRY_BREAKING",
            "content": rule_content,
            "source": "MiniASI量子坍縮決策",
            "generated_from": collision_points,
            "symmetry_level": symmetry_level,
            "strength": round(rule_strength, 2),
            "timestamp": datetime.now().isoformat(),
            "acceptance_score": 0.0,
            "conflict_keywords": self._extract_conflict_keywords(rule_content)
        }
        
        # 檢查規則衝突
        if not self._is_rule_conflicting(new_rule):
            self.civilization_rules.append(new_rule)
            print(f"📜 從對稱性衝突生成新規則：{new_rule['rule_id']} (強度: {rule_strength:.2f})")
            return new_rule
        else:
            print(f"⚠️ 跳過衝突規則生成：{rule_content}")
            return None
    
    def _generate_new_rule(
        self,
        entity1: CivilizationEntity,
        entity2: CivilizationEntity,
        collision_points: List[str],
        symmetry_context: bool = False
    ) -> Optional[Dict[str, Any]]:
        """從碰撞中生成新規則 v2.1"""
        
        # 檢查碰撞是否足夠重要
        if len(collision_points) < 1:
            return None
        
        # 根據碰撞類型決定規則類型
        rule_type = "EVOLUTIONARY"
        if symmetry_context or any("對稱" in point for point in collision_points):
            rule_type = "SYMMETRY_RESOLUTION"
        elif any("勇氣" in point for point in collision_points):
            rule_type = "COURAGE_BALANCE"
        elif any("量子" in point for point in collision_points):
            rule_type = "QUANTUM_CONSENSUS"
        elif any("穩定" in point for point in collision_points):
            rule_type = "STABILITY_PROTOCOL"
        
        # 創建新規則
        new_rule = {
            "rule_id": f"RULE_{rule_type[:3]}_{len(self.civilization_rules)+1:03d}",
            "type": rule_type,
            "content": f"處理碰撞點：{', '.join(collision_points[:2])}",
            "source": f"量子交互：{entity1.name} ↔ {entity2.name}",
            "generated_from": collision_points,
            "strength": 0.7,
            "timestamp": datetime.now().isoformat(),
            "acceptance_score": 0.0,
            "conflict_keywords": self._extract_conflict_keywords(collision_points[0] if collision_points else ""),
            "courage_context": [entity1.courage_level, entity2.courage_level] if "勇氣" in str(collision_points) else None,
            "stability_context": [entity1.stability_score, entity2.stability_score] if "穩定" in str(collision_points) else None
        }
        
        # 檢查規則衝突
        if self._is_rule_conflicting(new_rule):
            print(f"⚠️ 跳過衝突規則：{new_rule['content']}")
            return None
        
        # 添加到文明規則
        self.civilization_rules.append(new_rule)
        
        # 添加到實體的學習規則
        entity1.learned_rules.append(new_rule)
        entity2.learned_rules.append(new_rule)
        
        print(f"📜 生成新文明規則：{new_rule['rule_id']}")
        print(f"   內容：{new_rule['content']}")
        print(f"   類型：{rule_type}")
        print(f"   來源：{entity1.name} ↔ {entity2.name}")
        
        return new_rule
    
    def _record_interaction_attempt(
        self,
        entity1: CivilizationEntity,
        entity2: Optional[CivilizationEntity],
        topic: str,
        status: str,
        reason: str
    ) -> Dict[str, Any]:
        """記錄交互嘗試（無論成功與否）"""
        
        attempt_record = {
            "attempt_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "participants": [entity1.entity_id, entity2.entity_id if entity2 else "N/A"],
            "participant_names": [entity1.name, entity2.name if entity2 else "N/A"],
            "topic": topic,
            "status": status,
            "reason": reason,
            "entity1_state": {
                "energy": round(entity1.energy_level, 1),
                "quantum_state": round(entity1.quantum_state, 3),
                "courage": round(entity1.courage_level, 3),
                "stability": round(entity1.stability_score, 3),
                "entropy": round(entity1.get_current_entropy(), 3)
            },
            "entity2_state": {
                "energy": round(entity2.energy_level, 1) if entity2 else "N/A",
                "quantum_state": round(entity2.quantum_state, 3) if entity2 else "N/A",
                "courage": round(entity2.courage_level, 3) if entity2 else "N/A",
                "stability": round(entity2.stability_score, 3) if entity2 else "N/A",
                "entropy": round(entity2.get_current_entropy(), 3) if entity2 else "N/A"
            },
            "attempt_type": "INTERACTION_ATTEMPT",
            "cycle_number": self.evolution_cycles,
            "system_stability": round(self.system_stability, 3)
        }
        
        # 添加到交互嘗試日誌
        self.interaction_attempt_log.append(attempt_record)
        
        # 如果失敗，增加阻止計數
        if status in ["BLOCKED", "SAFETY_CHECK_FAILED", "NO_PARTNER_FOUND"]:
            self.blocked_interactions += 1
            self.system_stability = max(0.3, self.system_stability - 0.01)
        
        # 限制日誌長度
        if len(self.interaction_attempt_log) > 100:
            self.interaction_attempt_log = self.interaction_attempt_log[-100:]
        
        return attempt_record
    
    def run_evolution_cycle(self, miniasi_decision: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """運行一個演化週期 v2.1 - 量子演化（完全修復版）"""
        self.evolution_cycles += 1
        
        print(f"\n🔄 文明演化週期 #{self.evolution_cycles} (量子版本)")
        print(f"   系統穩定性：{self.system_stability:.2f}")
        
        cycle_results = {
            "cycle_number": self.evolution_cycles,
            "timestamp": datetime.now().isoformat(),
            "active_entities": len([e for e in self.entities.values() if e.can_interact()]),
            "interactions_attempted": 0,
            "interactions_successful": 0,
            "interactions_blocked": 0,
            "new_rules_generated": 0,
            "quantum_collapses": 0,
            "symmetry_resolutions": 0,
            "courage_adjustments": 0,
            "miniasi_decisions_processed": 0,
            "status": "COMPLETED",
            "interaction_details": [],
            "attempt_records": [],  # 新增：記錄所有嘗試
            "quantum_field_strength": self.superposition_field,
            "system_stability_before": round(self.system_stability, 3)
        }
        
        # 更新全局量子疊加場
        if self.entities:
            avg_quantum_state = sum(e.quantum_state for e in self.entities.values()) / len(self.entities)
            self.superposition_field = avg_quantum_state
        
        # 處理MiniASI決策（如果提供）
        if miniasi_decision:
            process_result = self.process_miniasi_decision(miniasi_decision)
            cycle_results["miniasi_decisions_processed"] = 1
            cycle_results["decision_processing"] = process_result
            
            if process_result.get("status") == "RULE_GENERATED":
                cycle_results["new_rules_generated"] += 1
                if process_result.get("symmetry_detected"):
                    cycle_results["symmetry_resolutions"] += 1
        
        # 為每個有能量的實體尋找交互機會
        for entity_id, entity in self.entities.items():
            # 記錄週期開始時的狀態
            initial_energy = entity.energy_level
            initial_quantum = entity.quantum_state
            
            if not entity.can_interact():
                # 記錄無法交互的原因
                readiness = entity.get_interaction_readiness()
                reason_parts = []
                if not readiness["energy_ok"]:
                    reason_parts.append(f"能量不足({entity.energy_level:.1f})")
                if not readiness["quantum_stable"]:
                    reason_parts.append(f"量子態異常({entity.quantum_state:.2f})")
                if not readiness["stability_ok"]:
                    reason_parts.append(f"穩定性低({entity.stability_score:.2f})")
                
                reason = " | ".join(reason_parts) if reason_parts else "未知原因"
                
                cycle_results["attempt_records"].append({
                    "entity": entity.name,
                    "reason": reason,
                    "status": "UNABLE_TO_ACT",
                    "details": readiness
                })
                continue
            
            cycle_results["interactions_attempted"] += 1
            
            # 尋找交互夥伴
            partner = self.find_interaction_partner(entity, self.superposition_field)
            if not partner:
                # 記錄找不到夥伴的具體原因
                available_others = [e for e in self.entities.values() 
                                  if e.entity_id != entity.entity_id and e.can_interact()]
                
                reason = "未知"
                if not available_others:
                    reason = "無其他可用實體"
                else:
                    # 檢查是否因為近期交互限制被過濾
                    recent_partners = entity.interaction_partners[-3:]
                    all_recent = all(other.entity_id in recent_partners for other in available_others)
                    if all_recent:
                        reason = "所有潛在夥伴都在近期交互列表中"
                    else:
                        reason = "匹配度計算結果為空 (檢查量子兼容性與勇氣平衡)"
                
                attempt_record = self._record_interaction_attempt(
                    entity, None, "尋找交互夥伴", 
                    "NO_PARTNER_FOUND", reason
                )
                cycle_results["attempt_records"].append(attempt_record)
                continue
            
            # 執行交互前檢查
            safety_check = self._safety_check_interaction(entity, partner)
            if not safety_check["safe"]:
                cycle_results["interactions_blocked"] += 1
                
                attempt_record = self._record_interaction_attempt(
                    entity, partner, 
                    self._generate_interaction_topic(entity, partner),
                    "SAFETY_CHECK_FAILED", " | ".join(safety_check["reasons"])
                )
                cycle_results["attempt_records"].append(attempt_record)
                continue
            
            # 執行交互
            topic = self._generate_interaction_topic(entity, partner)
            interaction = self.entity_interaction(entity, partner, topic)
            
            interaction_status = interaction.get("status", "UNKNOWN")
            if interaction_status not in ["FAILED", "BLOCKED"]:
                cycle_results["interactions_successful"] += 1
                
                if interaction.get("new_rule_generated", False):
                    cycle_results["new_rules_generated"] += 1
                
                if interaction.get("quantum_entangled", False):
                    cycle_results["quantum_collapses"] += 1
                
                if interaction.get("symmetry_context", False):
                    cycle_results["symmetry_resolutions"] += 1
                
                if any("勇氣" in point for point in interaction.get("collision_points", [])):
                    cycle_results["courage_adjustments"] += 1
            else:
                cycle_results["interactions_blocked"] += 1
            
            cycle_results["interaction_details"].append({
                "entity1": entity.name,
                "entity2": partner.name,
                "result": interaction_status,
                "new_rule": interaction.get("new_rule_generated", False),
                "quantum_entangled": interaction.get("quantum_entangled", False),
                "symmetry_context": interaction.get("symmetry_context", False),
                "energy_cost": interaction.get("energy_cost", 0)
            })
        
        # 能量恢復和量子狀態演化 (修復版：邏輯優化)
        for entity in self.entities.values():
            # 只有能量未滿時才恢復，防止數值溢出
            if entity.energy_level < 99.9:
                entity.gain_energy(5.0)
            
            # 量子狀態自然演化（趨向平衡）
            if entity.quantum_state > 0.6:
                # 過度不確定趨向穩定
                entity.quantum_state = max(0.3, entity.quantum_state - 0.05) 
            elif entity.quantum_state < 0.4:
                # 過度穩定趨向適度不確定
                entity.quantum_state = min(0.6, entity.quantum_state + 0.05)
            
            # 勇氣水平回歸 (防止長期極端化)
            if entity.courage_level > 0.8:
                entity.adjust_courage(-0.01)
            elif entity.courage_level < 0.2:
                entity.adjust_courage(0.01)
            
            # 穩定性自然恢復
            if entity.stability_score < 0.9:
                entity.stability_score = min(1.0, entity.stability_score + 0.02)
        
        # 系統穩定性更新
        success_ratio = (cycle_results["interactions_successful"] / 
                        max(1, cycle_results["interactions_attempted"]))
        
        if success_ratio > 0.5:
            self.system_stability = min(1.0, self.system_stability + 0.05)
        elif success_ratio < 0.2:
            self.system_stability = max(0.3, self.system_stability - 0.05)
        
        cycle_results["system_stability_after"] = round(self.system_stability, 3)
        cycle_results["success_ratio"] = round(success_ratio, 3)
        cycle_results["total_blocked_interactions"] = self.blocked_interactions
        
        print(f"   嘗試交互：{cycle_results['interactions_attempted']}次")
        print(f"   被阻止：{cycle_results['interactions_blocked']}次")
        print(f"   成功交互：{cycle_results['interactions_successful']}次")
        print(f"   成功比率：{success_ratio:.2f}")
        print(f"   生成新規則：{cycle_results['new_rules_generated']}條")
        print(f"   量子坍縮事件：{cycle_results['quantum_collapses']}次")
        print(f"   對稱性解決：{cycle_results['symmetry_resolutions']}次")
        print(f"   勇氣調整：{cycle_results['courage_adjustments']}次")
        print(f"   全局量子場：{self.superposition_field:.3f}")
        print(f"   系統穩定性：{self.system_stability:.3f} (+{cycle_results['system_stability_after'] - cycle_results['system_stability_before']:.3f})")
        
        return cycle_results
    
    def _generate_interaction_topic(self, entity1: CivilizationEntity, entity2: CivilizationEntity) -> str:
        """生成交互話題 v2.1 - 考慮穩定性"""
        base_topics = [
            "責任邊界的量子協調",
            "視野差異的疊加態整合",
            "熵值管理的糾纏策略",
            "規則演化的量子路徑",
            "能量分配的坍縮優化",
            "交互效率的量子提升",
            "文明穩定性的不確定性管理",
            "演化速度的量子控制"
        ]
        
        quantum_topics = [
            "量子糾纏下的責任共識",
            "疊加態價值權重平衡",
            "不確定性邊界的探索",
            "量子坍縮的決策優化"
        ]
        
        stability_topics = [
            "穩定性維護的量子策略",
            "系統平衡的糾纏方法",
            "風險規避的量子路徑",
            "安全交互的疊加態協議"
        ]
        
        # 根據實體特點選擇話題
        if entity1.quantum_state > 0.7 or entity2.quantum_state > 0.7:
            return random.choice(quantum_topics)
        elif entity1.stability_score < 0.5 or entity2.stability_score < 0.5:
            return random.choice(stability_topics)
        elif "邏輯" in entity1.responsibility or "邏輯" in entity2.responsibility:
            return "邏輯一致性的量子維護"
        elif "價值" in entity1.responsibility or "價值" in entity2.responsibility:
            return "價值權重的量子平衡"
        elif entity1.courage_level > 0.7 or entity2.courage_level > 0.7:
            return "勇氣過剩的量子修正"
        else:
            return random.choice(base_topics)
    
    def get_civilization_status(self) -> Dict[str, Any]:
        """獲取文明狀態 v2.1"""
        total_energy = sum(entity.energy_level for entity in self.entities.values())
        avg_autonomy = sum(entity.autonomy_score for entity in self.entities.values()) / max(1, len(self.entities))
        avg_quantum_state = sum(entity.quantum_state for entity in self.entities.values()) / max(1, len(self.entities))
        avg_courage = sum(entity.courage_level for entity in self.entities.values()) / max(1, len(self.entities))
        avg_stability = sum(entity.stability_score for entity in self.entities.values()) / max(1, len(self.entities))
        
        # 檢測異常狀態實體
        courage_excess = [e for e in self.entities.values() if e.courage_level > 0.7]
        quantum_unstable = [e for e in self.entities.values() if e.quantum_state > 0.8]
        low_stability = [e for e in self.entities.values() if e.stability_score < 0.4]
        low_energy = [e for e in self.entities.values() if e.energy_level < 30.0]
        
        return {
            "status": "QUANTUM_ACTIVE",
            "total_entities": len(self.entities),
            "total_energy": round(total_energy, 1),
            "average_autonomy": round(avg_autonomy, 3),
            "average_quantum_state": round(avg_quantum_state, 3),
            "average_courage": round(avg_courage, 3),
            "average_stability": round(avg_stability, 3),
            "courage_excess_count": len(courage_excess),
            "quantum_unstable_count": len(quantum_unstable),
            "low_stability_count": len(low_stability),
            "low_energy_count": len(low_energy),
            "total_rules": len(self.civilization_rules),
            "total_interactions": len(self.interaction_history),
            "total_attempts": len(self.interaction_attempt_log),
            "blocked_interactions": self.blocked_interactions,
            "evolution_cycles": self.evolution_cycles,
            "defer_count": self.defer_count,
            "symmetry_threshold": self.symmetry_threshold,
            "entity_weights": self.entity_weights,
            "quantum_field": round(self.superposition_field, 3),
            "system_stability": round(self.system_stability, 3),
            "entity_summary": [
                {
                    "name": entity.name,
                    "energy": round(entity.energy_level, 1),
                    "autonomy": round(entity.autonomy_score, 3),
                    "quantum_state": round(entity.quantum_state, 3),
                    "courage": round(entity.courage_level, 3),
                    "stability": round(entity.stability_score, 3),
                    "contributions": entity.evolution_contributions,
                    "entangled_partners": len(entity.quantum_entangled_partners),
                    "interaction_attempts": entity.interaction_attempts,
                    "can_interact": entity.can_interact(),
                    "interaction_readiness": entity.get_interaction_readiness()
                }
                for entity in self.entities.values()
            ],
            "recent_rules": self.civilization_rules[-5:] if self.civilization_rules else [],
            "recent_attempts": self.interaction_attempt_log[-5:] if self.interaction_attempt_log else [],
            "courage_excess_entities": [e.name for e in courage_excess],
            "quantum_unstable_entities": [e.name for e in quantum_unstable],
            "low_stability_entities": [e.name for e in low_stability],
            "low_energy_entities": [e.name for e in low_energy]
        }
    
    def save_state(self, filepath: str = "runtime/civilization_quantum_state.json"):
        """保存文明狀態 v2.1"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {
            "timestamp": datetime.now().isoformat(),
            "status": "QUANTUM_SAVED",
            "entities": {eid: entity.to_dict() for eid, entity in self.entities.items()},
            "civilization_rules": self.civilization_rules,
            "interaction_history_count": len(self.interaction_history),
            "interaction_attempt_log_count": len(self.interaction_attempt_log),
            "evolution_cycles": self.evolution_cycles,
            "defer_count": self.defer_count,
            "entity_weights": self.entity_weights,
            "quantum_field": self.superposition_field,
            "system_stability": self.system_stability,
            "blocked_interactions": self.blocked_interactions,
            "recent_interactions": self.interaction_history[-10:] if self.interaction_history else [],
            "recent_attempts": self.interaction_attempt_log[-10:] if self.interaction_attempt_log else [],
            "miniasi_decisions_cached": len(self.decision_cache),
            "courage_correction_active": self.courage_correction_active,
            "symmetry_threshold": self.symmetry_threshold,
            "quantum_entanglement_network": self.quantum_entanglement_network
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        print(f"💾 量子文明狀態已保存到：{filepath}")
        print(f"   實體數量：{len(self.entities)}")
        print(f"   文明規則：{len(self.civilization_rules)}")
        print(f"   交互歷史：{len(self.interaction_history)}")
        print(f"   嘗試日誌：{len(self.interaction_attempt_log)}")
        print(f"   系統穩定性：{self.system_stability:.3f}")
        
        return state


# 測試函數 v2.1 - 量子演化測試
def test_quantum_civilization():
    """測試量子文明演化系統"""
    print("🧪 測試量子文明演化系統 v2.1...")
    
    config = {
        "energy_recovery_rate": 5.0,
        "interaction_cost": 15.0,
        "max_interactions_per_cycle": 10,
        "quantum_field_decay": 0.95,
        "symmetry_threshold": 0.05
    }
    
    engine = CivilizationEngine(config)
    
    # 創建三個量子實體（對應三個AI角色）
    entities_config = [
        {
            "entity_id": "ENTITY_ACTION",
            "name": "量子行動者",
            "responsibility": "提出可執行的量子方案",
            "fear_of_loss": "行動陷入經典陷阱",
            "hard_limits": ["不能違反量子邏輯", "必須可量子回滾"],
            "vision_boundary": "量子疊加態中的可行路徑",
            "quantum_state": 0.3  # 較為確定
        },
        {
            "entity_id": "ENTITY_NEGATION",
            "name": "量子質疑者",
            "responsibility": "執行量子層面的真理否定",
            "fear_of_loss": "陷入經典否定循環",
            "hard_limits": ["不能逃避量子否定", "必須考慮疊加態"],
            "vision_boundary": "量子真理的可證偽邊界",
            "quantum_state": 0.6  # 中等不確定
        },
        {
            "entity_id": "ENTITY_VALUE",
            "name": "量子裁決者",
            "responsibility": "平衡量子代價與有機價值",
            "fear_of_loss": "量子價值坍縮為零",
            "hard_limits": ["不能忽略量子不確定性", "必須考慮糾纏效應"],
            "vision_boundary": "量子不確定性中的價值平衡點",
            "quantum_state": 0.4  # 略為確定
        }
    ]
    
    print("\n🏛️ 創建量子文明實體...")
    for config in entities_config:
        engine.create_entity(config)
    
    # 模擬MiniASI決策輸入
    print("\n🔄 模擬MiniASI量子決策處理...")
    
    # 正常量子決策
    normal_decision = {
        "input": "量子糾纏的非定域性問題",
        "status": "EXECUTED",
        "entropy": 0.7,
        "symmetry_detected": False,
        "courage_adjustment": 1.0,
        "results": {
            "ACTION": "基於量子力學接受非定域性",
            "NEGATION": "根據AXIOM_001，質疑非定域性的哲學含義",
            "VALUE": "需要在量子確定性與哲學不確定性之間平衡"
        },
        "scores": {
            "ACTION": 0.85,
            "NEGATION": 0.72,
            "VALUE": 0.78
        }
    }
    
    # 對稱性拮抗決策
    symmetry_decision = {
        "input": "量子計算的實用性邊界",
        "status": "DEFERRED",
        "entropy": 0.01,
        "symmetry_detected": True,
        "courage_adjustment": 1.1,
        "results": {
            "ACTION": "量子計算具有革命性潛力，應大力發展",
            "NEGATION": "量子計算仍存在根本性限制，不應過度樂觀",
            "VALUE": "兩者觀點都有道理，難以決斷"
        },
        "scores": {
            "ACTION": 0.835,
            "NEGATION": 0.827,
            "VALUE": 0.5
        }
    }
    
    # 測試正常決策處理
    print("\n1. 處理正常量子決策:")
    result1 = engine.process_miniasi_decision(normal_decision)
    print(f"   結果: {result1.get('status', 'UNKNOWN')}")
    
    # 測試對稱性決策處理
    print("\n2. 處理對稱性拮抗決策:")
    result2 = engine.process_miniasi_decision(symmetry_decision)
    print(f"   結果: {result2.get('status', 'UNKNOWN')}")
    if result2.get('status') == 'RULE_GENERATED':
        print(f"   生成對稱性規則: {result2.get('rule', {}).get('rule_id', '未知')}")
    
    # 注入AXIOM_014（勇氣過剩修復）
    print("\n3. 注入AXIOM_014規則:")
    rule_014 = engine.inject_rule(
        "014",
        "勇氣過剩時，價值AI必須主動降低行動置信度，防止系統過度勇敢",
        "COURAGE_CORRECTION"
    )
    
    if rule_014:
        print(f"   成功注入規則: {rule_014['rule_id']}")
    
    # 運行量子演化週期
    print("\n🔄 運行量子演化週期...")
    for i in range(3):
        print(f"\n週期 {i+1}:")
        results = engine.run_evolution_cycle(symmetry_decision if i == 1 else normal_decision)
        print(f"   結果：{results['interactions_successful']}次成功交互，")
        print(f"         {results['interactions_blocked']}次被阻止，")
        print(f"         {results['new_rules_generated']}條新規則，")
        print(f"         {results['quantum_collapses']}次量子坍縮，")
        print(f"         {results['courage_adjustments']}次勇氣調整")
        print(f"        系統穩定性：{results['system_stability_after']:.3f}")
    
    # 獲取量子文明狀態
    print("\n📊 量子文明狀態報告：")
    status = engine.get_civilization_status()
    
    print(f"   狀態：{status.get('status', 'UNKNOWN')}")
    print(f"   實體數量：{status['total_entities']}")
    print(f"   總能量：{status['total_energy']}")
    print(f"   平均量子狀態：{status['average_quantum_state']:.3f}")
    print(f"   平均勇氣水平：{status['average_courage']:.3f}")
    print(f"   平均穩定性：{status['average_stability']:.3f}")
    print(f"   勇氣過剩實體：{status['courage_excess_count']}個")
    print(f"   量子不穩定實體：{status['quantum_unstable_count']}個")
    print(f"   低穩定性實體：{status['low_stability_count']}個")
    print(f"   文明規則數量：{status['total_rules']}")
    print(f"   演化週期：{status['evolution_cycles']}")
    print(f"   僵局計數：{status['defer_count']}")
    print(f"   全局量子場：{status['quantum_field']:.3f}")
    print(f"   系統穩定性：{status['system_stability']:.3f}")
    print(f"   被阻止交互總數：{status['blocked_interactions']}")
    
    print("\n👥 量子實體狀態：")
    for entity_summary in status["entity_summary"]:
        print(f"   {entity_summary['name']}: ")
        print(f"     能量={entity_summary['energy']}, 量子態={entity_summary['quantum_state']:.3f}")
        print(f"     勇氣={entity_summary['courage']:.3f}, 穩定性={entity_summary['stability']:.3f}")
        print(f"     貢獻={entity_summary['contributions']}, 糾纏夥伴={entity_summary['entangled_partners']}個")
        print(f"     嘗試次數={entity_summary['interaction_attempts']}, 可交互={entity_summary['can_interact']}")
    
    # 保存狀態
    print("\n💾 保存量子文明狀態...")
    engine.save_state("runtime/quantum_civilization_test_v2.1.json")
    
    print("\n🎯 測試完成！系統已具備：")
    print("   ✅ 對稱性拮抗檢測與修復")
    print("   ✅ 勇氣過剩量子修正(AXIOM_014)")
    print("   ✅ 連續僵局動態擾動(AXIOM_006)")
    print("   ✅ 量子糾纏交互網絡")
    print("   ✅ 價值優先裁決機制")
    print("   ✅ 多重安全檢查系統")
    print("   ✅ 完整交互嘗試記錄")
    print("   ✅ 系統穩定性追蹤")
    
    return engine


if __name__ == "__main__":
    test_quantum_civilization()