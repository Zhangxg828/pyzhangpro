import json
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime
from config import SMART_CONTRACT_CONFIG, setup_logger

logger = setup_logger('smart_contract_interactor')

try:
    from web3 import Web3
    from web3.contract import Contract
    from web3.exceptions import ContractLogicError, TimeExhausted
    from eth_account import Account
    from eth_account.messages import encode_defunct
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logger.warning("web3未安装，智能合约交互功能将受限")


@dataclass
class ContractTransaction:
    tx_hash: str
    contract_address: str
    method: str
    params: Dict[str, Any]
    gas_used: int
    status: str
    timestamp: float


@dataclass
class ContractEvent:
    event_name: str
    contract_address: str
    args: Dict[str, Any]
    block_number: int
    transaction_hash: str
    timestamp: float


class SmartContractInteractor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or SMART_CONTRACT_CONFIG
        self.enabled = self.config.get('enabled', False)
        self.network = self.config.get('network', 'mainnet')
        self.gas_limit = self.config.get('gas_limit', 300000)
        self.gas_price_gwei = self.config.get('gas_price_gwei', 20)
        self.max_gas_price_gwei = self.config.get('max_gas_price_gwei', 100)
        self.contract_address = self.config.get('contract_address', '')
        self.private_key = self.config.get('private_key', '')
        self.rpc_endpoint = self.config.get('rpc_endpoint', '')
        
        self.w3: Optional[Web3] = None
        self.account: Optional[Account] = None
        self.contracts: Dict[str, Contract] = {}
        self.transaction_history: List[ContractTransaction] = []
        self.event_history: List[ContractEvent] = []
        
        if self.enabled and WEB3_AVAILABLE:
            self._initialize_web3()
        
        logger.info(f"智能合约交互器初始化完成 - 网络: {self.network}, 启用: {self.enabled}")

    def _initialize_web3(self):
        try:
            if not self.rpc_endpoint:
                logger.error("RPC端点未配置")
                return
                
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_endpoint))
            
            if not self.w3.is_connected():
                logger.error("无法连接到RPC端点")
                return
                
            logger.info(f"成功连接到 {self.network} 网络")
            
            if self.private_key:
                self.account = Account.from_key(self.private_key)
                logger.info(f"账户地址: {self.account.address}")
                
            if self.contract_address:
                self.load_contract(self.contract_address)
                
        except Exception as e:
            logger.error(f"初始化Web3失败: {e}")

    def load_contract(self, contract_address: str, abi: Optional[List[Dict]] = None) -> bool:
        try:
            if not self.w3:
                logger.error("Web3未初始化")
                return False
                
            if not abi:
                logger.warning(f"未提供ABI，使用通用ABI")
                abi = self._get_generic_abi()
                
            contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(contract_address),
                abi=abi
            )
            
            self.contracts[contract_address] = contract
            logger.info(f"合约加载成功: {contract_address}")
            return True
            
        except Exception as e:
            logger.error(f"加载合约失败: {e}")
            return False

    def _get_generic_abi(self) -> List[Dict]:
        return [
            {
                "inputs": [],
                "name": "getData",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "uint256", "name": "value", "type": "uint256"}],
                "name": "setData",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "address", "name": "from", "type": "address"},
                    {"indexed": False, "internalType": "uint256", "name": "value", "type": "uint256"}
                ],
                "name": "DataUpdated",
                "type": "event"
            }
        ]

    def call_contract_method(self, contract_address: str, method_name: str, 
                           params: Optional[Dict] = None, 
                           is_write: bool = False) -> Optional[Any]:
        try:
            if not self.w3 or contract_address not in self.contracts:
                logger.error(f"合约未加载: {contract_address}")
                return None
                
            contract = self.contracts[contract_address]
            method = getattr(contract.functions, method_name)
            
            if is_write:
                return self._execute_transaction(contract_address, method, params or {})
            else:
                if params:
                    result = method(**params).call()
                else:
                    result = method().call()
                    
                logger.info(f"调用合约方法成功: {method_name}, 结果: {result}")
                return result
                
        except Exception as e:
            logger.error(f"调用合约方法失败: {e}")
            return None

    def _execute_transaction(self, contract_address: str, method, params: Dict) -> Optional[str]:
        try:
            if not self.account:
                logger.error("账户未配置")
                return None
                
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            gas_price = self.w3.eth.gas_price
            gas_price_gwei = self.w3.from_wei(gas_price, 'gwei')
            
            if gas_price_gwei > self.max_gas_price_gwei:
                logger.warning(f"Gas价格过高: {gas_price_gwei:.2f} Gwei")
                return None
                
            transaction = method(**params).build_transaction({
                'from': self.account.address,
                'gas': self.gas_limit,
                'gasPrice': gas_price,
                'nonce': nonce
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash.hex(), timeout=120)
            
            tx_record = ContractTransaction(
                tx_hash=tx_hash.hex(),
                contract_address=contract_address,
                method=method.__name__,
                params=params,
                gas_used=receipt.gasUsed,
                status='success' if receipt.status == 1 else 'failed',
                timestamp=datetime.now().timestamp()
            )
            
            self.transaction_history.append(tx_record)
            
            logger.info(f"交易执行成功: {tx_hash.hex()}, Gas使用: {receipt.gasUsed}")
            
            return tx_hash.hex()
            
        except ContractLogicError as e:
            logger.error(f"合约逻辑错误: {e}")
            return None
        except TimeExhausted:
            logger.error("交易超时")
            return None
        except Exception as e:
            logger.error(f"执行交易失败: {e}")
            return None

    def deploy_contract(self, bytecode: str, abi: List[Dict], 
                       constructor_args: Optional[Dict] = None) -> Optional[str]:
        try:
            if not self.w3 or not self.account:
                logger.error("Web3或账户未初始化")
                return None
                
            contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)
            
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            gas_price = self.w3.eth.gas_price
            
            transaction = contract.constructor(**(constructor_args or {})).build_transaction({
                'from': self.account.address,
                'gas': self.gas_limit,
                'gasPrice': gas_price,
                'nonce': nonce
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash.hex(), timeout=120)
            
            contract_address = receipt.contractAddress
            
            logger.info(f"合约部署成功: {contract_address}")
            
            return contract_address
            
        except Exception as e:
            logger.error(f"部署合约失败: {e}")
            return None

    def listen_to_events(self, contract_address: str, event_name: str, 
                        from_block: Optional[int] = None) -> List[ContractEvent]:
        try:
            if not self.w3 or contract_address not in self.contracts:
                logger.error(f"合约未加载: {contract_address}")
                return []
                
            contract = self.contracts[contract_address]
            event = getattr(contract.events, event_name)
            
            if from_block is None:
                from_block = self.w3.eth.block_number - 1000
                
            events = event.get_logs(fromBlock=from_block)
            
            event_records = []
            for event_data in events:
                event_record = ContractEvent(
                    event_name=event_name,
                    contract_address=contract_address,
                    args=dict(event_data.args),
                    block_number=event_data.blockNumber,
                    transaction_hash=event_data.transactionHash.hex(),
                    timestamp=datetime.now().timestamp()
                )
                event_records.append(event_record)
                
            self.event_history.extend(event_records)
            
            logger.info(f"监听到 {len(event_records)} 个 {event_name} 事件")
            
            return event_records
            
        except Exception as e:
            logger.error(f"监听事件失败: {e}")
            return []

    def get_contract_balance(self, contract_address: str) -> Optional[int]:
        try:
            if not self.w3:
                logger.error("Web3未初始化")
                return None
                
            balance = self.w3.eth.get_balance(Web3.to_checksum_address(contract_address))
            return balance
            
        except Exception as e:
            logger.error(f"获取合约余额失败: {e}")
            return None

    def estimate_gas(self, contract_address: str, method_name: str, 
                    params: Optional[Dict] = None) -> Optional[int]:
        try:
            if not self.w3 or contract_address not in self.contracts:
                logger.error(f"合约未加载: {contract_address}")
                return None
                
            contract = self.contracts[contract_address]
            method = getattr(contract.functions, method_name)
            
            if params:
                gas_estimate = method(**params).estimate_gas({'from': self.account.address})
            else:
                gas_estimate = method().estimate_gas({'from': self.account.address})
                
            logger.info(f"Gas估算: {gas_estimate}")
            
            return gas_estimate
            
        except Exception as e:
            logger.error(f"估算Gas失败: {e}")
            return None

    def get_transaction_status(self, tx_hash: str) -> Optional[Dict]:
        try:
            if not self.w3:
                logger.error("Web3未初始化")
                return None
                
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            
            status = {
                'block_number': receipt.blockNumber,
                'gas_used': receipt.gasUsed,
                'status': 'success' if receipt.status == 1 else 'failed',
                'contract_address': receipt.contractAddress,
                'logs': receipt.logs
            }
            
            return status
            
        except Exception as e:
            logger.error(f"获取交易状态失败: {e}")
            return None

    def sign_message(self, message: str) -> Optional[str]:
        try:
            if not self.account:
                logger.error("账户未配置")
                return None
                
            message_hash = encode_defunct(text=message)
            signed_message = self.w3.eth.account.sign_message(message_hash, self.private_key)
            
            logger.info(f"消息签名成功: {message}")
            
            return signed_message.signature.hex()
            
        except Exception as e:
            logger.error(f"签名消息失败: {e}")
            return None

    def verify_message(self, message: str, signature: str, address: str) -> bool:
        try:
            if not self.w3:
                logger.error("Web3未初始化")
                return False
                
            message_hash = encode_defunct(text=message)
            recovered_address = self.w3.eth.account.recover_message(message_hash, signature=signature)
            
            is_valid = recovered_address.lower() == address.lower()
            
            logger.info(f"消息验证: {is_valid}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"验证消息失败: {e}")
            return False

    def get_network_info(self) -> Optional[Dict]:
        try:
            if not self.w3:
                logger.error("Web3未初始化")
                return None
                
            info = {
                'chain_id': self.w3.eth.chain_id,
                'block_number': self.w3.eth.block_number,
                'gas_price': self.w3.eth.gas_price,
                'network_name': self.network
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取网络信息失败: {e}")
            return None

    def get_transaction_history(self, contract_address: Optional[str] = None) -> List[ContractTransaction]:
        try:
            if contract_address:
                return [tx for tx in self.transaction_history 
                       if tx.contract_address == contract_address]
            return self.transaction_history
        except Exception as e:
            logger.error(f"获取交易历史失败: {e}")
            return []

    def get_event_history(self, contract_address: Optional[str] = None, 
                         event_name: Optional[str] = None) -> List[ContractEvent]:
        try:
            events = self.event_history
            
            if contract_address:
                events = [e for e in events if e.contract_address == contract_address]
                
            if event_name:
                events = [e for e in events if e.event_name == event_name]
                
            return events
        except Exception as e:
            logger.error(f"获取事件历史失败: {e}")
            return []

    def batch_call_methods(self, calls: List[Dict]) -> List[Optional[Any]]:
        try:
            results = []
            for call in calls:
                result = self.call_contract_method(
                    call['contract_address'],
                    call['method_name'],
                    call.get('params'),
                    call.get('is_write', False)
                )
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"批量调用失败: {e}")
            return []