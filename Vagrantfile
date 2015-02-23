Vagrant.configure('2') do |config|
  vm_ram = ENV['VAGRANT_VM_RAM'] || 2560
  vm_cpu = ENV['VAGRANT_VM_CPU'] || 2

  config.vm.box = "precise64"
  config.vm.box_url = 'http://files.vagrantup.com/precise64.box'

  config.vm.provider :virtualbox do |vb|
    vb.customize ["modifyvm", :id, "--memory", vm_ram, "--cpus", vm_cpu]
  end
  
  config.vm.provision :shell, :inline => "sudo -u vagrant /vagrant/init.sh"

end
