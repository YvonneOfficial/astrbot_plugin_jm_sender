from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import astrbot.api.message_components as Comp
import jmcomic
import os
import re
import asyncio
from typing import List, Optional
import img2pdf
try:
    from PIL import Image
except ImportError:
    Image = None
import aiohttp
import shutil
from collections import defaultdict
import datetime
import time

@register("astrbot-jm-sender", "AstrBot User", "下载禁漫并以群消息转发的形式发送图片", "1.0.0")
class JmSender(Star):
    """
    JM漫画下载并转发发送器
    
    使用方法：
    /jms <漫画ID> - 下载指定漫画并以转发消息形式发送所有图片
    /jmsp <漫画ID> <章节ID> - 下载指定漫画的指定章节并发送图片
    """
    
    def __init__(self, context: Context):
        super().__init__(context)
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.option_path = os.path.join(self.current_dir, 'option.yml')
        self.download_dir = os.path.join(self.current_dir, 'data', 'jmcomic_files')
        os.makedirs(self.download_dir, exist_ok=True)
        
        # 初始化JM选项
        try:
            self.option = jmcomic.JmOption.from_file(self.option_path)
            # logger.error(self.option.)
            # 创建JM客户端
            self.client = self.option.new_jm_client()
            logger.info(f"JM漫画插件初始化成功，下载目录: {self.download_dir}")

            # 启动定时清理任务
            self._start_cleanup_scheduler()
        except Exception as e:
            logger.error(f"JM漫画插件初始化失败: {e}")
            self.option = None
            self.client = None
            
    def _start_cleanup_scheduler(self):
        """启动定时清理任务"""
        asyncio.create_task(self._daily_cleanup_task())
        logger.info("已启动每日漫画文件清理定时任务")
    
    async def _daily_cleanup_task(self):
        """每天凌晨3点清理下载目录"""
        while True:
            try:
                # 计算距离下一个凌晨3点的时间
                now = datetime.datetime.now()
                next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)
                if now >= next_run:
                    # 如果当前时间已经过了今天的3点，计算到明天3点的时间
                    next_run = next_run + datetime.timedelta(days=1)
                
                # 计算等待时间
                wait_seconds = (next_run - now).total_seconds()
                logger.info(f"下一次漫画文件清理将在 {next_run.strftime('%Y-%m-%d %H:%M:%S')} 进行，等待 {wait_seconds:.2f} 秒")
                
                # 等待到指定时间
                await asyncio.sleep(wait_seconds)
                
                # 执行清理
                await self._cleanup_comic_files()
                
                # 等待一小段时间，避免连续执行
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"定时清理任务出错: {e}")
                # 如果出错，等待一段时间后重试
                await asyncio.sleep(3600)  # 1小时后重试
    
    async def _cleanup_comic_files(self):
        """清理下载目录中的漫画文件"""
        try:
            if os.path.exists(self.download_dir):
                logger.info(f"开始执行每日漫画文件清理: {self.download_dir}")
                
                # 统计清理前的文件数量和大小
                total_size_before = self._get_dir_size(self.download_dir)
                
                # 执行清理操作
                shutil.rmtree(self.download_dir)
                os.makedirs(self.download_dir, exist_ok=True)
                
                logger.info(f"漫画文件清理完成，释放了 {total_size_before / (1024*1024):.2f} MB 空间")
                return True
        except Exception as e:
            logger.error(f"清理漫画文件时出错: {e}")
            return False
    
    def _get_dir_size(self, path):
        """获取目录大小（字节）"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        return total_size

    def _should_use_pdf(self, event: AstrMessageEvent) -> bool:
        """判断是否使用PDF形式发送"""
        platform_name = event.get_platform_name()
        if platform_name != "aiocqhttp":
            return False

        # 尝试根据事件提供的接口判断是否为群聊
        for attr in ("is_group", "is_group_message", "is_group_event"):
            if hasattr(event, attr):
                flag = getattr(event, attr)
                if callable(flag):
                    try:
                        return bool(flag())
                    except Exception:
                        continue
                return bool(flag)

        # OneBot v11 兼容处理
        for attr in ("message_type", "detail_type", "context_type"):
            value = getattr(event, attr, None)
            if isinstance(value, str) and value.lower() == "group":
                return True

        # 无法判断则默认走PDF，避免触发风控
        return True

    def _sanitize_filename(self, name: str) -> str:
        """清理文件名中的非法字符"""
        sanitized = re.sub(r'[\\\\/:*?"<>|]', "_", name)
        sanitized = sanitized.strip() or "未命名章节"
        return sanitized

    def _create_pdf_from_images(self, photo_dir: str, image_files: List[str], chapter_title: str) -> Optional[str]:
        """将章节图片合并为PDF并返回文件路径"""
        try:
            safe_title = self._sanitize_filename(chapter_title)
            pdf_dir = os.path.join(self.download_dir, "pdf_exports")
            os.makedirs(pdf_dir, exist_ok=True)
            pdf_path = os.path.join(pdf_dir, f"{safe_title}.pdf")
            img_paths = [os.path.join(photo_dir, f) for f in image_files]

            with open(pdf_path, "wb") as pdf_fp:
                pdf_fp.write(img2pdf.convert(img_paths))

            return pdf_path
        except Exception as primary_error:
            logger.warning(f"img2pdf转换失败，尝试Pillow回退: {primary_error}")
            if Image is None:
                logger.error("未安装Pillow，无法进行PDF回退转换")
                return None

            try:
                images = []
                for img_path in img_paths:
                    with Image.open(img_path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        images.append(img.copy())

                if not images:
                    return None

                images[0].save(pdf_path, save_all=True, append_images=images[1:])
                for img in images:
                    img.close()
                return pdf_path
            except Exception as fallback_error:
                logger.error(f"使用Pillow合并PDF失败: {fallback_error}")
                return None

    def _build_pdf_component(self, pdf_path: str):
        """根据运行环境生成PDF组件"""
        if not os.path.exists(pdf_path):
            return None

        pdf_name = os.path.basename(pdf_path)

        file_component = None
        file_class = getattr(Comp, "File", None)
        if file_class is not None:
            builder = getattr(file_class, "fromFileSystem", None)
            if callable(builder):
                try:
                    file_component = builder(pdf_path, name=pdf_name)
                except Exception as e:
                    logger.error(f"创建File组件失败: {e}")
            else:
                try:
                    file_component = file_class(path=pdf_path, name=pdf_name)
                except Exception as e:
                    logger.error(f"实例化File组件失败: {e}")

        if file_component is None:
            logger.warning("无法找到可用的文件消息组件，PDF将不会发送")

        return file_component
    
    @filter.command("jms")
    async def download_and_send(self, event: AstrMessageEvent):
        """下载整本漫画并发送"""
        # 解析漫画ID
        message = event.message_str
        match = re.search(r'\d+', message)
        if not match:
            event.set_result(MessageEventResult().message("请提供正确的漫画ID，例如: /jms 12345"))
            return
            
        album_id = match.group(0)
        
        # 发送下载开始提示
        yield event.chain_result([
            Comp.Plain(f"开始下载漫画 {album_id}，请稍候...")
        ])
        
        try:
            # 下载漫画
            result = await self._download_album(album_id)
            if not result:
                event.set_result(MessageEventResult().message(f"下载漫画 {album_id} 失败"))
                return
                
            album_detail, all_photos = result
            
            # 发送漫画信息
            yield event.chain_result([
                Comp.Plain(f"漫画《{album_detail.name}》下载完成，共 {len(all_photos)} 个章节，即将开始发送...")
            ])
            
            # 按章节组织并发送图片
            for photo in all_photos:
                # 使用async for来迭代异步生成器
                async for message in self._send_photo_images(event, photo, album_detail.name):
                    yield message
                # 避免发送过快
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"下载并发送漫画时出错: {e}")
            event.set_result(MessageEventResult().message(f"发送漫画时出错: {str(e)}"))
            
    @filter.command("jmsp")
    async def download_and_send_photo(self, event: AstrMessageEvent):
        """下载指定章节并发送"""
        # 解析参数
        message = event.message_str
        match = re.findall(r'\d+', message)
        if len(match) < 2:
            event.set_result(MessageEventResult().message("请提供正确的漫画ID和章节ID，例如: /jmsp 12345 67890"))
            return
            
        album_id = match[0]
        photo_id = match[1]
        
        # 发送下载开始提示
        yield event.chain_result([
            Comp.Plain(f"开始下载漫画 {album_id} 的章节 {photo_id}，请稍候...")
        ])
        
        try:
            # 下载章节
            photo = await self._download_photo(photo_id)
            if not photo:
                event.set_result(MessageEventResult().message(f"下载章节 {photo_id} 失败"))
                return
                
            # 发送章节信息
            yield event.chain_result([
                Comp.Plain(f"章节《{photo.title}》下载完成，共 {len(photo.image_list)} 张图片，即将开始发送...")
            ])
            
            # 发送章节图片，使用async for来迭代异步生成器
            async for message in self._send_photo_images(event, photo, photo.title):
                yield message
                
        except Exception as e:
            logger.error(f"下载并发送章节时出错: {e}")
            event.set_result(MessageEventResult().message(f"发送章节时出错: {str(e)}"))
    
    async def _download_album(self, album_id):
        """下载整本漫画"""
        if not self.option or not self.client:
            logger.error("JM选项初始化失败，无法下载")
            return None
            
        try:
            # 获取漫画信息
            album_detail = self.client.get_album_detail(album_id)
            
            # 下载漫画
            logger.info(f"开始下载漫画: {album_id}")
            jmcomic.download_album(album_id, self.option)
            
            # 获取所有章节详情
            all_photos = []
            for photo in album_detail:
                photo_detail = self.client.get_photo_detail(photo.photo_id)
                if photo_detail:
                    all_photos.append(photo_detail)
                    
            return album_detail, all_photos
            
        except Exception as e:
            logger.error(f"下载漫画 {album_id} 时出错: {e}")
            return None
    
    async def _download_photo(self, photo_id):
        """下载单个章节"""
        if not self.option or not self.client:
            logger.error("JM选项初始化失败，无法下载")
            return None
            
        try:
            # 获取章节信息
            photo_detail = self.client.get_photo_detail(photo_id)
            
            # 下载章节所有图片
            logger.info(f"开始下载章节: {photo_id}")
            jmcomic.download_photo(photo_id, self.option)
            
            return photo_detail
        except Exception as e:
            logger.error(f"下载章节 {photo_id} 时出错: {e}")
            return None
    
    async def _send_photo_images(self, event: AstrMessageEvent, photo, title: Optional[str] = None):
        """以转发消息的形式发送章节的所有图片"""
        try:
            chapter_title = title or getattr(photo, "title", "未命名章节")
            # 准备转发消息节点
            # 根据实际目录结构构建照片目录路径
            # 首先尝试预期的目录结构：album_id/photo_id
            photo_dir = os.path.join(self.download_dir, str(photo.album_id), str(photo.photo_id))
            
            # 如果预期目录不存在，尝试直接使用album_id目录
            if not os.path.exists(photo_dir):
                logger.info(f"尝试使用备用目录结构: {photo_dir}不存在，尝试直接使用album_id目录")
                photo_dir = os.path.join(self.download_dir, str(photo.album_id))
                
                # 如果备用目录也不存在，则报错
                if not os.path.exists(photo_dir):
                    logger.error(f"所有可能的章节目录都不存在: {photo_dir}")
                    return
                
            logger.info(f"使用目录: {photo_dir}来获取图片")
                
            # 获取所有图片路径并排序
            image_files = sorted([f for f in os.listdir(photo_dir) if f.endswith(('.jpg', '.png', '.webp', '.jpeg'))])
            if not image_files:
                logger.error(f"章节目录中没有找到图片: {photo_dir}")
                return
            
            logger.info(f"在{photo_dir}中找到{len(image_files)}张图片")
            
            # 获取平台信息并决定发送策略
            platform_name = event.get_platform_name()
            use_pdf = self._should_use_pdf(event)
            pdf_path = None
            if use_pdf:
                pdf_dir = os.path.join(self.download_dir, "pdf_exports")
                existing_pdf = os.path.join(pdf_dir, f"{self._sanitize_filename(chapter_title)}.pdf")
                if os.path.exists(existing_pdf):
                    pdf_path = existing_pdf
                else:
                    pdf_path = await asyncio.to_thread(
                        self._create_pdf_from_images,
                        photo_dir,
                        image_files,
                        chapter_title
                    )
                if pdf_path:
                    pdf_component = self._build_pdf_component(pdf_path)
                    if pdf_component:
                        yield event.chain_result([
                            Comp.Plain(f"章节《{chapter_title}》已打包为PDF，共{len(image_files)}页，请查收。"),
                            pdf_component
                        ])
                        return
                    else:
                        logger.warning("生成PDF成功但无法构建消息组件，回退到普通转发发送")
                else:
                    logger.warning("生成PDF失败，将回退到普通转发发送")
                # 若PDF生成失败，则继续使用原来的转发逻辑
                use_pdf = False

            # 无论是私聊还是群聊，都使用转发消息
            # 构建转发消息节点
            nodes = []

            for i, img_file in enumerate(image_files):
                img_path = os.path.join(photo_dir, img_file)

                # 根据不同平台设置不同的发送者信息
                if platform_name == "webchat":
                    # webchat平台使用字符串作为uin，使用Nodes可能不适用于webchat
                    # 尝试直接发送图片而不使用Node
                    if i %29 ==0 :
                        yield event.chain_result([
                            Comp.Plain(chapter_title)
                        ])

                    yield event.chain_result([
                        Comp.Plain(f"第 {i+1}/{len(image_files)} 页\n"),
                        Comp.Image.fromFileSystem(img_path)
                    ])
                    # 图片发送后等待一小段时间
                    await asyncio.sleep(0.5)
                    continue
                else:
                    # 其他平台如QQ等使用转发消息
                    # Extract a usable sender uin
                    self_id = event.get_self_id()
                    uin_str = None
                    if self_id is not None:
                        digit_match = re.search(r'\d+', str(self_id))
                        if digit_match:
                            uin_str = digit_match.group(0)
                    if not uin_str:
                        uin_str = str(self_id or "0")
                        logger.warning(f"Platform {platform_name} self_id '{self_id}' cannot be parsed as digits, using {uin_str} as fallback uin")
                    try:
                        uin_value = int(uin_str)
                    except (TypeError, ValueError):
                        uin_value = 10000
                        logger.warning(f"Fallback uin_value=10000 for platform {platform_name} due to invalid uin_str: {uin_str}")
                    else:
                        if uin_value <= 0:
                            uin_value = 10000
                            logger.warning(f"Non-positive uin_value derived ({uin_str}); using fallback 10000")
                    
                    # 添加到节点列表
                    if i %29 ==0 :
                        nodes.append(
                            Comp.Node(
                                name="AstrBot",
                                uin=uin_value,
                                content=[
                                    Comp.Plain(chapter_title)
                                ]
                            )
                        )

                    nodes.append(
                        Comp.Node(
                            name="AstrBot",
                            uin=uin_value,
                            content=[
                                Comp.Plain(f"第 {i+1}/{len(image_files)} 页\n"),
                                Comp.Image.fromFileSystem(img_path)
                            ]
                        )
                    )
            
            # 如果是支持合并转发的平台且nodes不为空，则用合并转发发送
            if nodes:
                # 每次最多发送30张图片，避免消息过大
                batch_size = 30
                for i in range(0, len(nodes), batch_size):
                    batch_nodes = nodes[i:i+batch_size]
                    if batch_nodes:
                        # 发送这一批次的图片
                        try:
                            yield event.chain_result([
                                Comp.Nodes(batch_nodes)
                            ])
                            # 等待一下再发送下一批
                            await asyncio.sleep(2)
                        except Exception as e:
                            logger.error(f"发送合并转发消息失败: {e}，尝试使用普通方式发送")
                            # 如果转发消息发送失败，尝试直接发送
                            for node in batch_nodes:
                                try:
                                    content = node.content
                                    yield event.chain_result(content)
                                    await asyncio.sleep(1)
                                except Exception as node_error:
                                    logger.error(f"发送单个节点消息失败: {node_error}")
                    
        except Exception as e:
            logger.error(f"发送章节图片时出错: {e}")
            
    async def terminate(self):
        """清理资源"""
        # 清理下载的文件，避免占用太多空间
        try:
            if os.path.exists(self.download_dir):
                # 有选择地保留一部分数据或全部清理
                logger.info(f"清理下载目录: {self.download_dir}")
                # 在此启用清理功能
                shutil.rmtree(self.download_dir)
                os.makedirs(self.download_dir, exist_ok=True)
                logger.info(f"下载目录已清理完成")
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")
    
    @filter.command("jmclean")
    async def manual_cleanup(self, event: AstrMessageEvent):
        """手动清理下载目录"""
        yield event.chain_result([
            Comp.Plain("开始清理漫画下载目录，请稍候...")
        ])
        
        try:
            # 执行清理
            result = await self._cleanup_comic_files()
            
            if result:
                yield event.chain_result([
                    Comp.Plain("漫画下载目录清理完成！所有漫画文件已被删除")
                ])
            else:
                yield event.chain_result([
                    Comp.Plain("漫画下载目录清理失败，请检查日志")
                ])
        except Exception as e:
            logger.error(f"手动清理出错: {e}")
            yield event.chain_result([
                Comp.Plain(f"清理时出错: {str(e)}")
            ]) 
