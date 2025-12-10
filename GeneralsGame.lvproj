<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="25008000">
	<Property Name="NI.LV.All.SaveVersion" Type="Str">25.0</Property>
	<Property Name="NI.LV.All.SourceOnly" Type="Bool">true</Property>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="client" Type="Folder">
			<Item Name="drawsquare.vi" Type="VI" URL="../client/drawsquare.vi"/>
			<Item Name="pregame.vi" Type="VI" URL="../client/pregame.vi"/>
		</Item>
		<Item Name="common" Type="Folder">
			<Item Name="types" Type="Folder">
				<Item Name="CommandCluster.ctl" Type="VI" URL="../common/types/CommandCluster.ctl"/>
				<Item Name="LobbyPlayerInfoArray.ctl" Type="VI" URL="../common/types/LobbyPlayerInfoArray.ctl"/>
			</Item>
			<Item Name="ClientNetworkInterface.vi" Type="VI" URL="../common/ClientNetworkInterface.vi"/>
			<Item Name="ClientNetworkInterfaceGlobalVariables.vi" Type="VI" URL="../common/ClientNetworkInterfaceGlobalVariables.vi"/>
		</Item>
		<Item Name="server" Type="Folder">
			<Item Name="DataBroadcaster.vi" Type="VI" URL="../server/DataBroadcaster.vi"/>
			<Item Name="FieldBroadcaster.vi" Type="VI" URL="../server/FieldBroadcaster.vi"/>
			<Item Name="IsValidCoordinate.vi" Type="VI" URL="../server/IsValidCoordinate.vi"/>
			<Item Name="LobbyBroadcaster.vi" Type="VI" URL="../server/LobbyBroadcaster.vi"/>
			<Item Name="Server.vi" Type="VI" URL="../server/Server.vi"/>
			<Item Name="ServerGlobalVariables.vi" Type="VI" URL="../server/ServerGlobalVariables.vi"/>
			<Item Name="ServerNetworkInterface.vi" Type="VI" URL="../server/ServerNetworkInterface.vi"/>
			<Item Name="ServerNetworkInterfaceGlobalVariables.vi" Type="VI" URL="../server/ServerNetworkInterfaceGlobalVariables.vi"/>
		</Item>
		<Item Name="tests" Type="Folder">
			<Item Name="PlayerJoinTest.vi" Type="VI" URL="../tests/PlayerJoinTest.vi"/>
			<Item Name="PlayerLeaveTest.vi" Type="VI" URL="../tests/PlayerLeaveTest.vi"/>
			<Item Name="SetOptionTest.vi" Type="VI" URL="../tests/SetOptionTest.vi"/>
			<Item Name="SetReadyTest.vi" Type="VI" URL="../tests/SetReadyTest.vi"/>
			<Item Name="SetTeamTest.vi" Type="VI" URL="../tests/SetTeamTest.vi"/>
		</Item>
		<Item Name="Dependencies" Type="Dependencies"/>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
