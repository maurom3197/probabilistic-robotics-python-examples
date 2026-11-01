const MODULE_LABELS={
  Motion_Models:'Motion Models',Sensors_Models:'Sensor Models',
  Gaussian_Filters:'Gaussian Filters',Discrete_Filters:'Discrete Filters',
  SLAM:'SLAM',Mapping:'Mapping','2D_maps':'2D Maps',
};
const $=id=>document.getElementById(id);

async function loadSource(){
  const response=await fetch('./source/manifest.json',{cache:'no-store'});
  if(!response.ok)throw new Error(`Source manifest could not be loaded (HTTP ${response.status}).`);
  const manifest=await response.json();
  if(!/^[0-9a-f]{40}$/.test(manifest.commit)||!Array.isArray(manifest.files))throw new Error('Invalid source manifest.');
  const commit=document.createElement('a');
  commit.href=`https://github.com/maurom3197/probabilistic-robotics-python-examples/commit/${manifest.commit}`;
  commit.target='_blank';commit.rel='noopener';commit.textContent=manifest.commit.slice(0,12)+' ↗';
  $('source-commit').replaceChildren('Main commit: ',commit);
  $('source-file-count').textContent=`${manifest.files.length} FILES`;
  const groups=new Map();
  for(const file of manifest.files){
    if(!groups.has(file.module))groups.set(file.module,[]);
    groups.get(file.module).push(file);
  }
  for(const module of [...manifest.modules,'2D_maps']){
    const files=groups.get(module)||[];
    const section=document.createElement('section');section.className='card source-module';
    const heading=document.createElement('h2');heading.textContent=MODULE_LABELS[module]||module;
    const count=document.createElement('span');count.className='small-label';count.textContent=`${files.length} FILES`;
    const head=document.createElement('div');head.className='section-title';head.append(heading,count);section.append(head);
    const list=document.createElement('ul');
    for(const file of files){
      const item=document.createElement('li'),link=document.createElement('a');
      const url='./source/'+file.path.split('/').map(encodeURIComponent).join('/');
      link.href=url;link.textContent=file.path.slice(module.length+1);
      if(file.path.endsWith('.py'))link.addEventListener('click',async event=>{
        event.preventDefault();
        $('preview-title').textContent=file.path;
        $('preview-content').textContent='Loading…';
        $('preview-download').href=url;$('preview-download').hidden=false;
        const result=await fetch(url,{cache:'no-store'});
        $('preview-content').textContent=result.ok?await result.text():`Could not load file (HTTP ${result.status}).`;
        $('preview-content').scrollTop=0;
        document.querySelector('.source-preview').scrollIntoView({behavior:'smooth',block:'start'});
      });
      item.append(link);list.append(item);
    }
    section.append(list);$('source-modules').append(section);
  }
}
loadSource().catch(error=>{$('source-commit').textContent=error.message;});
