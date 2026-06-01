from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"

#验证用户输入的密码是否正确
#输入：用户输入的密码(明文)、数据库中存储的密码哈希值
#输出：布尔值，表示密码是否匹配
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

#生成密码的哈希值
#输入：用户输入的密码(明文)
#输出：密码的哈希值
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

#创建访问令牌
#输入：包含用户信息的字典(例如：{"sub": str(user.id)})，可选的过期时间
#输出：访问令牌字符串
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    # 计算过期时间  datetime.utcnow() 返回当前时间的 UTC 时间
    expire = datetime.utcnow() + (
        # 如果提供了过期时间，就使用它，否则，就使用默认的过期时间(7天)
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)

#解码访问令牌
#输入：访问令牌字符串
#输出：包含用户信息的字典(例如：{"sub": str(user.id)})，如果令牌无效，返回None
def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
